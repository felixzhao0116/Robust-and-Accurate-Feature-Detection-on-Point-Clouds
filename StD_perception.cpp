#include "StD_perception.h"
#include <algorithm>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/centroid.h>
#include <cmath>

#include <boost/thread/thread.hpp>
#include "OrientNormal.h"
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>

#include <pcl/visualization/pcl_visualizer.h>

static bool compareIntensity(const pcl::PointXYZINormal& a, const pcl::PointXYZINormal& b) {
	return a.intensity > b.intensity;
}

static void splitPointNormal(const pcl::PointCloud<pcl::PointXYZINormal>& cloud_in,
	pcl::PointCloud<pcl::PointXYZI>& cloud_out_xyz,
	pcl::PointCloud<pcl::Normal>& cloud_out_normal)
{
	// Resize output clouds
	cloud_out_xyz.width = cloud_in.width;
	cloud_out_xyz.height = cloud_in.height;
	cloud_out_xyz.is_dense = cloud_in.is_dense;
	cloud_out_xyz.points.resize(cloud_in.size());

	cloud_out_normal.width = cloud_in.width;
	cloud_out_normal.height = cloud_in.height;
	cloud_out_normal.is_dense = cloud_in.is_dense;
	cloud_out_normal.points.resize(cloud_in.size());

	for (size_t i = 0; i < cloud_in.points.size(); i++)
	{
		cloud_out_xyz.points[i].x = cloud_in.points[i].x;
		cloud_out_xyz.points[i].y = cloud_in.points[i].y;
		cloud_out_xyz.points[i].z = cloud_in.points[i].z;
		cloud_out_xyz.points[i].intensity = cloud_in.points[i].intensity;

		cloud_out_normal.points[i].normal_x = cloud_in.points[i].normal_x;
		cloud_out_normal.points[i].normal_y = cloud_in.points[i].normal_y;
		cloud_out_normal.points[i].normal_z = cloud_in.points[i].normal_z;
		cloud_out_normal.points[i].curvature = cloud_in.points[i].curvature;
	}
}

std::vector<std::vector<int>> StD_perception::splitNeighbors(const int& p_indice,
													const std::vector<int>& nn_indices) {
	// Compute the centroid
	Eigen::Vector3f centroid(0, 0, 0);
	for (const int& i : nn_indices) {
		centroid += potential_feat_->points[nn_indices[i]].getVector3fMap();
	}
	centroid /= static_cast<float>(nn_indices.size());

	std::vector<std::vector<int>> result;
	for (const int& i : nn_indices) {
		std::vector<int> subNbrA, subNbrB;

		// Define the plane
		Eigen::Vector3f normal = (centroid - (*potential_feat_)[p_indice].getVector3fMap()).cross(potential_feat_->points[nn_indices[i]].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap());
		if (normal.norm() < 1e-6) continue;//It implies that these three points are collinear and cannot generate a plane.
		normal.normalize();

		for (const int& j : nn_indices) {
			if (normal.dot(potential_feat_->points[nn_indices[j]].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap()) > 0) {
				subNbrA.push_back(j);
			}
			if (normal.dot(potential_feat_->points[nn_indices[j]].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap()) < 0) {
				subNbrB.push_back(j);
			}
			if (normal.dot(potential_feat_->points[nn_indices[j]].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap()) == 0) {
				subNbrA.push_back(j);
				subNbrB.push_back(j);
			}
		}
		result.push_back(subNbrA);
		result.push_back(subNbrB);
	}
	return result;
}
float StD_perception::calWCPMetric(const int p_indice, const std::vector<int> nn_indices) {
	size_t n = nn_indices.size();

	//计算平面参数
	Eigen::Vector4f plane_parameters;
	plane_parameters[0] = (*normal_)[p_indice].normal_x;
	plane_parameters[1] = (*normal_)[p_indice].normal_y;
	plane_parameters[2] = (*normal_)[p_indice].normal_z;
	plane_parameters[3] = 0;
	plane_parameters[3] = -1 * plane_parameters.dot((*potential_feat_)[p_indice].getVector4fMap());

	//计算邻域点的两个metric
	Eigen::Vector4f centroid;
	Eigen::Vector3f ni = plane_parameters.block<3, 1>(0, 0);
	pcl::compute3DCentroid<PointT>(*potential_feat_, nn_indices, centroid);
	centroid[3] = 0;
	float d_sum = 0, theta_sum=0;
	std::vector<float> d(n);
	std::vector<float> theta(n);
	for (size_t j = 0; j < n; j++) {
		d[j]=fabs((centroid - (*potential_feat_)[nn_indices[j]].getVector4fMap()).dot(plane_parameters));
		d_sum += d[j];
		
		Eigen::Vector3f nj = { (*normal_)[nn_indices[j]].normal_x,(*normal_)[nn_indices[j]].normal_y ,(*normal_)[nn_indices[j]].normal_z };
		ni.normalize();
		nj.normalize();
		double dot_product = ni.dot(nj);
		dot_product = std::min(1.0, std::max(-1.0, dot_product)); // Due to floating-point inaccuracies, ni.dot(nj) might slightly exceed 1. So you need to clamp it.
		theta[j] = fabs(acos(dot_product));
		theta_sum += theta[j];
	}
	//计算两个metric的均值
	float theta_mean, d_mean;
	theta_mean = theta_sum / n;
	d_mean = d_sum / n;
	//计算rou和pi
	float rou = 0, pi = 0;
	for (size_t j = 0; j < n; j++) {
		rou += pow(theta[j] - theta_mean, 2);
		pi += pow(d[j] - d_mean, 2);
	}
	rou = sqrt(rou / n);
	pi = sqrt(pi / n);
	//计算kn kp
	float kn = exp(rou / mu_n_);
	float kp = exp(pi / mu_p_);

	//计算D
	float D = fabs((centroid.block<3,1>(0,0) - (*potential_feat_)[p_indice].getVector3fMap()).dot(ni));

	return (kn * kp * D);
}

PointCloudPtr StD_perception::detectFeaturePoints() {
	//normal calculation
	pcl::NormalEstimationOMP < PointT, pcl::Normal > ne;
	ne.setInputCloud(input_);
	ne.setSearchMethod(tree_);
	ne.setNumberOfThreads(64);
	ne.setRadiusSearch(radius_);
	ne.compute(*normal_);

	//normal orientation propogation
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr input_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
	pcl::concatenateFields(*input_, *normal_, *input_with_normals);
	OrientNormal origen;
	origen.consistentTangentPlane(input_with_normals, 20, false); //***
	splitPointNormal(*input_with_normals, *input_, *normal_);

	float delta = 0;	//weighted centroid projection metric
	int z = 0;//the number of neighborhood splitting operation
	std::vector<std::vector<int>> potential_feat_nn_indices;
	*potential_feat_ = *input_;
	*potential_feat_normal_ = *normal_;

	while (feat_->size() < static_cast<size_t>(alpha_ * input_->size()) || z < 3) {
		//Calculate the variation metric pointwisely
		for (int p_indice = 0; p_indice < potential_feat_->size(); p_indice++) {
			//calculate process
			if (z == 0) { //Initialize the neighborhood
				std::vector<int> nn_indices;
				std::vector<float> nn_dists;
				tree_->radiusSearch((*input_)[p_indice], radius_, nn_indices, nn_dists);
				potential_feat_nn_indices.push_back(nn_indices);
				delta = calWCPMetric(p_indice, nn_indices);
				potential_feat_->points[p_indice].intensity = delta;
			}
			else {
				//split the neighborhood and get 2K sub-neighborhoods
				std::vector<std::vector<int>> sub_neighborhoods;
				sub_neighborhoods = splitNeighbors(p_indice, potential_feat_nn_indices[p_indice]);
				//select the neighborhood with the largest metric
				double maxWCP = -std::numeric_limits<double>::infinity();
				std::vector<int> bestSubNbr;
				for (const auto& sn : sub_neighborhoods) {
					double currentWCP = calWCPMetric(p_indice, sn);
					if (currentWCP > maxWCP) {
						maxWCP = currentWCP;
						bestSubNbr = sn;
					}
				}
				potential_feat_nn_indices[p_indice] = bestSubNbr;
				potential_feat_->points[p_indice].intensity = delta;
				z++;
			}
			
		}
		//Sort the potential feature point cloud(V) based on the variation metric.
		pcl::PointCloud<pcl::PointXYZINormal>::Ptr potential_feat_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
		pcl::concatenateFields(*potential_feat_, *potential_feat_normal_, *potential_feat_with_normals);
		std::sort(potential_feat_with_normals->points.begin(), potential_feat_with_normals->points.end(), compareIntensity);
		size_t topN = static_cast<size_t>(potential_feat_with_normals->size() * beta_);
		size_t bottomM = static_cast<size_t>(potential_feat_with_normals->size() * gamma_);
		splitPointNormal(*potential_feat_with_normals, *potential_feat_, *potential_feat_normal_);

		//Update data
		feat_->points.insert(feat_->points.end(), potential_feat_->points.begin(), potential_feat_->points.begin() + topN);
		potential_feat_->points.assign(potential_feat_->points.begin() + topN + 1, potential_feat_->points.end() - bottomM);
		potential_feat_normal_->points.assign(potential_feat_normal_->points.begin() + topN + 1, potential_feat_normal_->points.end() - bottomM);
		potential_feat_nn_indices.assign(potential_feat_nn_indices.begin() + topN + 1, potential_feat_nn_indices.end() - bottomM);
	
	}

	return feat_;

}
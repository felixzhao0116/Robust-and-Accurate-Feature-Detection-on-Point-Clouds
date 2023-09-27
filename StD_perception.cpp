#include "StD_perception.h"
#include <algorithm>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/centroid.h>
#include <cmath>

#include <boost/thread/thread.hpp>
#include "OrientNormal.h"
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>

#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <thread>
#include <chrono>

#include <pcl/visualization/pcl_visualizer.h>

static bool compareIntensity(const PointT& a, const PointT& b) {
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
		centroid += input_->points[i].getVector3fMap();
	}
	centroid /= static_cast<float>(nn_indices.size());

	std::vector<std::vector<int>> result;
	for (const int& i : nn_indices) {
		std::vector<int> subNbrA, subNbrB;

		// Define the plane
		Eigen::Vector3f normal = (centroid - (*potential_feat_)[p_indice].getVector3fMap()).cross(potential_feat_->points[i].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap());

		if (normal.norm() < 1e-6) continue;//It implies that these three points are collinear and cannot generate a plane.
		normal.normalize();

		for (const int& j : nn_indices) {
			if (normal.dot(input_->points[j].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap()) > 0) {
				subNbrA.push_back(j);
			}
			if (normal.dot(input_->points[j].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap()) < 0) {
				subNbrB.push_back(j);
			}
			if (normal.dot(input_->points[j].getVector3fMap() - (*potential_feat_)[p_indice].getVector3fMap()) == 0) {
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
	plane_parameters[0] = (*potential_feat_normal_)[p_indice].normal_x;
	plane_parameters[1] = (*potential_feat_normal_)[p_indice].normal_y;
	plane_parameters[2] = (*potential_feat_normal_)[p_indice].normal_z;
	plane_parameters[3] = 0;
	plane_parameters[3] = -1 * plane_parameters.dot((*potential_feat_)[p_indice].getVector4fMap());

	//计算邻域点的两个metric
	Eigen::Vector4f centroid;
	Eigen::Vector3f ni = plane_parameters.block<3, 1>(0, 0);
	pcl::compute3DCentroid<PointT>(*input_, nn_indices, centroid);
	centroid[3] = 0;
	float d_sum = 0, theta_sum=0;
	std::vector<float> d(n);
	std::vector<float> theta(n);
	for (size_t j = 0; j < n; j++) {
		d[j]=fabs((centroid - (*input_)[nn_indices[j]].getVector4fMap()).dot(plane_parameters));
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

void StD_perception::updatePotentialFeature() {
	std::vector<size_t> idx(potential_feat_->size());
	std::iota(idx.begin(), idx.end(), 0);  // 初始化为[0, 1, 2, ...]
	std::sort(idx.begin(), idx.end(),
		[this](size_t i1, size_t i2) {
			return compareIntensity(potential_feat_->points[i1], potential_feat_->points[i2]);
		});


	auto old_potential_feat = *potential_feat_;
	auto old_potential_feat_normal = *potential_feat_normal_;
	auto old_potential_feat_nn_indices = potential_feat_nn_indices;

	for (size_t i = 0; i < idx.size(); ++i) {
		potential_feat_->points[i] = old_potential_feat.points[idx[i]];
		potential_feat_normal_->points[i] = old_potential_feat_normal.points[idx[i]];
		potential_feat_nn_indices[i] = old_potential_feat_nn_indices[idx[i]];
	}

	size_t topN = static_cast<size_t>(potential_feat_->size() * beta_);
	size_t bottomM = static_cast<size_t>(potential_feat_->size() * gamma_);

	feat_->points.insert(feat_->points.end(), potential_feat_->points.begin(), potential_feat_->points.begin() + topN);
	potential_feat_->points.assign(potential_feat_->points.begin() + topN + 1, potential_feat_->points.end() - bottomM);
	potential_feat_normal_->points.assign(potential_feat_normal_->points.begin() + topN + 1, potential_feat_normal_->points.end() - bottomM);
	potential_feat_nn_indices.assign(potential_feat_nn_indices.begin() + topN + 1, potential_feat_nn_indices.end() - bottomM);
	std::cout << "========The feature points propotion is " + std::to_string(static_cast<float>(feat_->size()) / input_->size() * 100) + "%========" << std::endl;
}

PointCloudPtr StD_perception::detectFeaturePoints() {
	//normal calculation
	std::cout << "========Calculate the normal========" << std::endl;
	tt.tic();
	pcl::NormalEstimationOMP < PointT, pcl::Normal > ne;
	ne.setInputCloud(input_);
	ne.setSearchMethod(tree_);
	ne.setNumberOfThreads(64);
	ne.setRadiusSearch(radius_);
	ne.compute(*normal_);
	std::cout << "The execution time: " << tt.toc() << "ms" << std::endl << std::endl;

	//normal orientation propogation
	std::cout << "========Orient the normal========" << std::endl;
	tt.tic();
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr input_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
	pcl::concatenateFields(*input_, *normal_, *input_with_normals);
	OrientNormal origen;
	origen.consistentTangentPlane(input_with_normals, 20, false); //***
	splitPointNormal(*input_with_normals, *input_, *normal_);
	std::cout << "The execution time: " << tt.toc() << "ms" << std::endl;
	std::cout << "The normal is visualized. Close the visualizer to continue." << std::endl << std::endl;
	
	//The visualization of the normal
	boost::shared_ptr<pcl::visualization::PCLVisualizer> oriented_normal_viewer(new pcl::visualization::PCLVisualizer("oriented_normal_viewer"));
	oriented_normal_viewer->setWindowName("oriented_normal_viewer");
	oriented_normal_viewer->addText("oriented_normal_viewer", 50, 50, 0, 1, 0, "v1_text");
	oriented_normal_viewer->addPointCloud<pcl::PointXYZINormal>(input_with_normals, "input_with_normals");
	oriented_normal_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "input_with_normals");
	oriented_normal_viewer->addPointCloudNormals<pcl::PointXYZINormal>(input_with_normals, 1, 10, "normals");

	while (!oriented_normal_viewer->wasStopped())
	{
		oriented_normal_viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	float delta = 0;	//weighted centroid projection metric
	int z = 0;//the number of neighborhood splitting operation

	*potential_feat_ = *input_;
	*potential_feat_normal_ = *normal_;
	
	while (feat_->size() < static_cast<size_t>(alpha_ * input_->size()) && z < splitting_n_) {
		if (z == 0) {
			std::cout << "======Initialize the neighborhood========" << std::endl;
			tt.tic();
		}
		else std::cout << "======Split the neighborhood: " + std::to_string(z)+" ========" << std::endl;

		using namespace indicators;
		show_console_cursor(false);
		indicators::ProgressBar bar{
		  option::BarWidth{50},
		  option::Start{" ["},
		  option::Fill{"█"},
		  option::Lead{"█"},
		  option::Remainder{"-"},
		  option::End{"]"},
		  option::PrefixText{"Detecting the feature points "},
		  option::ForegroundColor{Color::yellow},
		  option::ShowElapsedTime{true},
		  option::ShowRemainingTime{true},
		  option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
		};
		
		//Calculate the variation metric pointwisely
		int step = potential_feat_->size() / 10;
		int progress = 0;
		for (int p_indice = 0; p_indice < potential_feat_->size(); p_indice++) {
			//calculate process
			if (z == 0) { //Initialize the neighborhood
				if (p_indice % step == 0 && progress <= 100) {
					bar.set_progress(progress);
					progress += 10;
				}
				std::vector<int> nn_indices;
				std::vector<float> nn_dists;
				tree_->radiusSearch((*input_)[p_indice], radius_, nn_indices, nn_dists);
				potential_feat_nn_indices.push_back(nn_indices);
				delta = calWCPMetric(p_indice, nn_indices);
				potential_feat_->points[p_indice].intensity = delta;
			}
			else {
				if (p_indice % step == 0 && progress <= 100) {
					bar.set_progress(progress);
					progress += 10;
				}
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
			}	
		}
		show_console_cursor(true);

		//Sort the potential feature point cloud(V) based on the variation metric.
		updatePotentialFeature();
		std::cout << "The execution time: " << tt.toc() << "ms" << std::endl << std::endl;
		z = z + 1;
	}

	return feat_;

}
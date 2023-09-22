#include "StD_perception.h"
#include <algorithm>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/centroid.h>
#include <cmath>

#include <pcl/visualization/pcl_visualizer.h>

static bool compareIntensity(const PointT& a, const PointT& b) {
	return a.intensity > b.intensity;
}

float StD_perception::calWCPMetric(const int indice) {
	//搜索p的邻域
	std::vector<int> nn_indices;
	std::vector<float> nn_dists;
	tree_->radiusSearch((*input_)[indice], radius_, nn_indices, nn_dists);
	size_t n = nn_indices.size();


	//计算平面参数
	Eigen::Vector4f plane_parameters;
	plane_parameters[0] = (*normal_)[indice].normal_x;
	plane_parameters[1] = (*normal_)[indice].normal_y;
	plane_parameters[2] = (*normal_)[indice].normal_z;
	plane_parameters[3] = 0;
	plane_parameters[3] = -1 * plane_parameters.dot((*input_)[indice].getVector4fMap());

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
	float D = fabs((centroid.block<3,1>(0,0) - (*input_)[indice].getVector3fMap()).dot(ni));

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

	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
	viewer.addPointCloud<PointT>(input_, "sample cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

	viewer.addPointCloudNormals<pcl::PointXYZI, pcl::Normal>(input_, normal_, 1, 0.005, "normals");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");
	//viewer.spin();

	//逐点计算variation metric \delta_i
	float delta = 0;	//weighted centroid projection metric

	for (int i = 0; i < input_->size(); i++) {
		//calculate process
		delta = calWCPMetric(i);
		input_->points[i].intensity = delta;

	}
	//排序\delta, 分类PF PnonF V
	PointCloudPtr potential_feat_(new PointCloud);
	std::sort(input_->points.begin(), input_->points.end(), compareIntensity);
	size_t topN = static_cast<size_t>(input_->size() * beta_);
	size_t bottomM = static_cast<size_t>(input_->size() * gamma_);
	feat_->points.assign(input_->begin(), input_->begin() + topN);
	potential_feat_->points.assign(input_->begin() + topN + 1, input_->end() - bottomM);

	return feat_;

	//对V做splitting and selection

}
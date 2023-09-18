#define PCL_NO_PRECOMPILE 1

#include "StD_perception.h"
#include <algorithm>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/centroid.h>
#include <cmath>

static bool compareIntensity(const PointT& a, const PointT& b) {
	return a.intensity > b.intensity;
}

float StD_perception::calWCPMetric(const PointT p) {
	////搜索p的邻域
	//std::vector<int> nn_indices;
	//std::vector<float> nn_dists;
	//tree_->radiusSearch(p, radius_, nn_indices, nn_dists);

	//size_t n = nn_indices.size();

	////计算平面参数
	//Eigen::Vector4f plane_parameters;
	//plane_parameters[0] = p.normal_x;
	//plane_parameters[1] = p.normal_y;
	//plane_parameters[2] = p.normal_z;
	//plane_parameters[3] = 0;
	//plane_parameters[3] = -1 * plane_parameters.dot(p.getVector4fMap());

	////计算邻域点的两个metric
	//Eigen::Vector4f centroid;
	//Eigen::Vector3f ni = plane_parameters.block<3, 1>(0, 0);
	//pcl::compute3DCentroid<PointT>(*input_, nn_indices, centroid);
	//centroid[3] = 0;
	//float d_sum = 0, theta_sum=0;
	//std::vector<float> d(n);
	//std::vector<float> theta(n);
	//for (size_t j = 0; j < n; j++) {
	//	PointT tmp = input_->points[nn_indices[j]];
	//	
	//	d.push_back((centroid - p.getVector4fMap()).dot(plane_parameters));
	//	d_sum += d[j];
	//	
	//	Eigen::Vector3f nj = { tmp.normal_x,tmp.normal_y ,tmp.normal_z };
	//	theta.push_back(acos(ni.normalized().dot(nj.normalized())) / M_PI * 180.0);
	//	theta_sum += theta[j];
	//}
	////计算两个metric的均值
	//float theta_mean, d_mean;
	//theta_mean = theta_sum / n;
	//d_mean = d_sum / n;
	////计算rou和pi
	//float rou = 0, pi = 0;
	//for (size_t j = 0; j < n; j++) {
	//	rou += pow(theta[j] - theta_mean, 2);
	//	pi += pow(d[j] - d_mean, 2);
	//}
	//rou = sqrt(rou / n);
	//pi = sqrt(pi / n);
	////计算kn kp
	//float kn = exp(rou / mu_n_);
	//float kp = exp(pi / mu_p_);

	////计算D
	//float D = (centroid - p.getVector3fMap()).transpose().dot(ni);

	return 0;
	//return (kn * kp * D);
}


PointCloudPtr StD_perception::detectFeaturePoints() {
	//计算法线
	pcl::NormalEstimationOMP<PointT, PointT> ne;
	ne.setInputCloud(input_);
	ne.setSearchMethod(tree_);
	ne.setNumberOfThreads(64);
	ne.setRadiusSearch(0.5);
	ne.compute(*input_);

	//逐点计算variation metric \delta_i
	float delta = 0;	//weighted centroid projection metric

	for (int i = 0; i < input_->size(); i++) {
		//calculate process
		delta = calWCPMetric(input_->points[i]);
		input_->points[i].intensity = delta;

	}
	//排序\delta, 分类PF PnonF V
	PointCloudPtr potential_feat_(new PointCloud);
	std::sort(input_->points.begin(), input_->points.end(), compareIntensity);
	size_t topN = static_cast<size_t>(input_->size() * beta_);
	size_t bottomM = static_cast<size_t>(input_->size() * gamma_);
	feat_->points.assign(input_->begin(), input_->begin() + topN);
	potential_feat_->points.assign(input_->begin() + topN + 1, input_->end() - bottomM);

	//对V做splitting and selection

}
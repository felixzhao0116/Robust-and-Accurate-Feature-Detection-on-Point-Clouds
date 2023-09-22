#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

using PointT = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = PointCloud::Ptr;
using SearchPtr = pcl::search::KdTree<PointT>::Ptr;

class StD_perception {
public:
	StD_perception()
		: beta_(0.2)
		, gamma_(0.2)
		, radius_(0.003)
		, mu_n_(30)
		, mu_p_(0.1)
	{
		normal_ = std::make_shared<pcl::PointCloud<pcl::Normal>>();
		feat_ = std::make_shared<PointCloud>();
	}

	inline void 
		setInputCloud(const PointCloudPtr& cloud) { input_ = cloud; }

	inline void 
		setUpperPotentialBound(double beta) { beta_ = beta; }

	inline double 
		getUpperPotentialBound() { return beta_; }

	inline void
		setLowerPotentialBound(double gamma) { gamma_ = gamma; }

	inline double
		getLowerPotentialBound() { return gamma_; }

	inline void
		setSearchTree(SearchPtr tree) { tree_ = tree; }

	inline void 
		setSearchRadius(float radius) { radius_ = radius; }

	inline void
		setScalingFactor(float mu_n, float mu_p) { mu_n_ = mu_n; mu_p_ = mu_p; }

	PointCloudPtr detectFeaturePoints();


private:
	float calWCPMetric(const int indice);

protected:
	//The input cloud
	PointCloudPtr input_;

	//The feature cloud
	PointCloudPtr feat_;

	//The search object for picking subsequent samples using radius search
	SearchPtr tree_;

	//Points ranked in the top beta% will be considered as feature points
	float beta_;

	//Points ranked in the bottom gamma% will be considered as non-feature points
	float gamma_;

	//The search radius
	float radius_;

	//The normal scaling factor mu_n
	float mu_n_;

	//The dist scaling factor mu_p
	float mu_p_;

	//The normal of the input point cloud
	pcl::PointCloud<pcl::Normal>::Ptr normal_;
};

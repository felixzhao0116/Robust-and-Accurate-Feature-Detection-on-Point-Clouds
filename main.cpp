#include <iostream>
#include <boost/program_options.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <cmath>

#include "StD_perception.h"


namespace po = boost::program_options;

int main(int argc, char *argv[]) {
	//Command line parameter
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("search_radius,r", po::value<int>(), "Please enter a multiplier for the search radius (between 5 and 10) ")
		("bilateral_weight_normal,n", po::value<double>(), "Please enter the scaling factor of the normal (between 0.07 and 0.25) ")
		("bilateral_weight_plane,p", po::value<double>(), "Please enter a multiplier for the scaling factor of the plane (between 0.2 and 0.8) ");
	po::variables_map vm;

	//load point cloud
	PointCloudPtr input_cloud(new PointCloud);//global input
	PointCloudPtr feat_cloud(new PointCloud);//global output
	pcl::io::loadPCDFile<PointT>("D:\\workspace\\Robust-and-Accurate-Feature-Detection-on-Point-Clouds\\pointcloud\\bun_zipper.pcd", *input_cloud);

	//calculate the average distance between each point and its nearest neighbor to generate the hyper-parameters
	SearchPtr tree_init(new pcl::search::KdTree<PointT>);
	tree_init->setInputCloud(input_cloud);
	std::vector<int> nearest_indice;
	std::vector<float> nearest_dist;
	double dist_sum;
	for (int i = 0; i < input_cloud->size(); i++) {
		tree_init->nearestKSearch((*input_cloud)[i], 2, nearest_indice, nearest_dist);//If you set k = 1, then the nearest point will be the query point itself.
		dist_sum += sqrt(nearest_dist[1]);
	}
	double dist_mean = dist_sum / input_cloud->size();
	std::cout << "The average distance between each point is " << dist_mean << std::endl;
	
	//Command line parameter parsing
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}

		if (vm.count("search_radius")) {
			std::cout << "The search_radius was set to " << vm["search_radius"].as<int>()* dist_mean << ".\n";
		}

		if (vm.count("bilateral_weight_normal")) {
			std::cout << "The scaling factor of normal was set to " << vm["bilateral_weight_normal"].as<double>() << ".\n";
		}

		if (vm.count("bilateral_weight_plane")) {
			std::cout << "The scaling factor of plane was set to " << vm["bilateral_weight_plane"].as<double>() * dist_mean << ".\n";
		}
	}
	catch (const po::error& ex) {
		std::cerr << "Error:" << ex.what() << "\n";
		std::cerr << desc << "\n";
		return -1;
	}

	//Structure-to-detail feature perception
	StD_perception detector;
	detector.setInputCloud(input_cloud);
	SearchPtr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(input_cloud);
	detector.setSearchTree(tree);
	detector.setSearchRadius(vm["search_radius"].as<int>() * dist_mean);
	detector.setUpperPotentialBound(0.08);
	detector.setLowerPotentialBound(0.25);
	detector.setScalingFactor(vm["bilateral_weight_normal"].as<double>(), vm["bilateral_weight_plane"].as<double>() * dist_mean);
	feat_cloud = detector.detectFeaturePoints();
	

	//特征点可视化，可供下游任务：点云分割、特征线提取、表面重建使用
	int v1 = 0;
	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addPointCloud(input_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(input_cloud, 179, 179, 179), "input_cloud", v1);
	viewer.addPointCloud(feat_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(feat_cloud, 0, 0, 255), "feat_cloud", v1);

	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "feat_cloud");

	while (!viewer.wasStopped()) {
		viewer.spinOnce(1, true);
	}
	
	return 0;
}
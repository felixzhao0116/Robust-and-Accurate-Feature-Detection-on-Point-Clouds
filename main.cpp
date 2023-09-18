#include <iostream>
#include <boost/program_options.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "StD_perception.h"

//这个后边研究一下
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>


namespace po = boost::program_options;

int main(int argc, char *argv[]) {
	//命令行参数解析
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("filename,f", po::value<std::string>(), "set filename")
		("optimization,o", po::value<int>()->default_value(10), "optimization level");

	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}

		if (vm.count("filename")) {
			std::cout << "Filename was set to " << vm["filename"].as<std::string>() << ".\n";
		}
		else {
			std::cout << "Filename was not set.\n";
		}

		std::cout << "Optimization level is " << vm["optimization"].as<int>() << ".\n";
	}
	catch (const po::error& ex) {
		std::cerr << "Error:" << ex.what() << "\n";
		std::cerr << desc << "\n";
		return -1;
	}

	//读取点云
	PointCloudPtr input_cloud(new PointCloud);//全局输入
	PointCloudPtr feat_cloud(new PointCloud);//全局输出
	pcl::io::loadPCDFile<PointT>("D:\\workspace\\Robust-and-Accurate-Feature-Detection-on-Point-Clouds\\pointcloud\\bun_zipper.pcd", *input_cloud);

	//Structure-to-detail feature perception
	StD_perception detector;
	detector.setInputCloud(input_cloud);
	SearchPtr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(input_cloud);
	detector.setSearchTree(tree);
	detector.setUpperPotentialBound(0.2);
	detector.setLowerPotentialBound(0.2);
	detector.setScalingFactor(30, 0.1);
	

	//特征点可视化，可供下游任务：点云分割、特征线提取、表面重建使用
	int v1 = 0;
	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addPointCloud(input_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(input_cloud, 0, 0, 0), "input_cloud", v1);
	viewer.addPointCloud(feat_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(feat_cloud, 255, 0, 0), "feat_cloud", v1);

	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "feat_cloud");

	while (!viewer.wasStopped()) {
		viewer.spinOnce(1, true);
	}
	
	return 0;
}
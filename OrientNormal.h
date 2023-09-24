#pragma once
#pragma once
#include <iostream>
#include <utility> // std::pair 
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/mst_orient_normals.h>      // 最小生成树法线定向

// Types
typedef CGAL::Simple_cartesian<float> Kernel;
// 定义存储点和法线的容器
typedef std::pair<Kernel::Point_3, Kernel::Vector_3> PointVectorPair;

class OrientNormal
{

public:
	OrientNormal() {}
	~OrientNormal() {}

	void consistentTangentPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pclNormal, const int neighbor, bool ereaseNon);

};



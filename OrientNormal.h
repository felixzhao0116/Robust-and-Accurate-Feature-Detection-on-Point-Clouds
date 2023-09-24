#pragma once
#pragma once
#include <iostream>
#include <utility> // std::pair 
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/mst_orient_normals.h>      // ��С���������߶���

// Types
typedef CGAL::Simple_cartesian<float> Kernel;
// ����洢��ͷ��ߵ�����
typedef std::pair<Kernel::Point_3, Kernel::Vector_3> PointVectorPair;

class OrientNormal
{

public:
	OrientNormal() {}
	~OrientNormal() {}

	void consistentTangentPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pclNormal, const int neighbor, bool ereaseNon);

};



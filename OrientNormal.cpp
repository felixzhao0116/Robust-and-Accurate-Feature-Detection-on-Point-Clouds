#include "OrientNormal.h"

// 最小代价生成树用于法向量定向
void OrientNormal::consistentTangentPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pclNormal,
	const int nb_neighbors = 20, bool ereaseNon = false)
{
	//pcl->CGAL
	std::vector<PointVectorPair> points;
	for (size_t i = 0; i < pclNormal->size(); ++i)
	{
		float px = pclNormal->points[i].x;
		float py = pclNormal->points[i].y;
		float pz = pclNormal->points[i].z;
		float nx = pclNormal->points[i].normal_x;
		float ny = pclNormal->points[i].normal_y;
		float nz = pclNormal->points[i].normal_z;
		points.push_back(PointVectorPair(Kernel::Point_3(px, py, pz), Kernel::Vector_3(nx, ny, nz)));
	}
	// ---------------------------------法线定向--------------------------------------
	//注意:mst_orient_normals()需要一个范围的点以及属性映射来访问每个点的位置和法线。
	auto unoriented_points_begin = CGAL::mst_orient_normals(points, nb_neighbors,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
		.normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

	if (ereaseNon == true)
	{
		// 可选操作: 删除未进行法线定向的点
		points.erase(unoriented_points_begin, points.end());
		if (pclNormal->size() != points.size())
		{
			//::cout << "原始点云有：" << pclNormal->size() << "个点" << std::endl;
			//std::cout << "定向之后的点云有：" << points.size() << "个点" << std::endl;
			//std::cout << "删除掉的未定向点数有：" << pclNormal->size() - points.size() << "个" << std::endl;
			pclNormal->resize(points.size());
		}
	}

	// ---------------------------------结果保存--------------------------------------
	for (size_t i = 0; i < points.size(); ++i)
	{
		pclNormal->points[i].x = points[i].first.hx();
		pclNormal->points[i].y = points[i].first.hy();
		pclNormal->points[i].z = points[i].first.hz();
		pclNormal->points[i].normal_x = points[i].second.hx();
		pclNormal->points[i].normal_y = points[i].second.hy();
		pclNormal->points[i].normal_z = points[i].second.hz();
	}

}

#include "OrientNormal.h"

// ��С�������������ڷ���������
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
	// ---------------------------------���߶���--------------------------------------
	//ע��:mst_orient_normals()��Ҫһ����Χ�ĵ��Լ�����ӳ��������ÿ�����λ�úͷ��ߡ�
	auto unoriented_points_begin = CGAL::mst_orient_normals(points, nb_neighbors,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
		.normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

	if (ereaseNon == true)
	{
		// ��ѡ����: ɾ��δ���з��߶���ĵ�
		points.erase(unoriented_points_begin, points.end());
		if (pclNormal->size() != points.size())
		{
			//::cout << "ԭʼ�����У�" << pclNormal->size() << "����" << std::endl;
			//std::cout << "����֮��ĵ����У�" << points.size() << "����" << std::endl;
			//std::cout << "ɾ������δ��������У�" << pclNormal->size() - points.size() << "��" << std::endl;
			pclNormal->resize(points.size());
		}
	}

	// ---------------------------------�������--------------------------------------
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

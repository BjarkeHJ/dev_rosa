#include <iostream>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>

#ifndef __ROSA_MAIN__
#define __ROSA_MAIN__

struct SkeletonDecomposition 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals;
    Eigen::MatrixXd pts_matrix;
    Eigen::MatrixXd nrs_matrix;
    std::vector<std::vector<int>> neighs;
};

class RosaMain {
public:
    /* Functions */
    void init(std::shared_ptr<rclcpp::Node> node);
    void main();

    /* Data */


    /* Utils */
    SkeletonDecomposition SSD;

private:
    /* Functions */
    void normalize();
    void mahanalobis_mat(double &radius_r);
    double pt_mahalanobis_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double &range_r);
    void drosa();
    void dcrosa();

    /* Params */
    int ne_KNN; // K normal estimation neighbours
    double radius_neigh; // Radius for mahanalobis neighbour search
    double th_mah; // Mahanalobis distance threshold

    /* Data */
    int pcd_size_;
    double norm_scale;
    Eigen::Vector4d centroid;

    /* Utils */
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne; // Normal estimation
    pcl::VoxelGrid<pcl::PointNormal> vgf; // Voxel Grid Filter for downsamplings

};

#endif //ROSA_MAIN
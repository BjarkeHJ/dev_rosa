#ifndef __GLOBAL_SKEL__
#define __GLOBAL_SKEL__

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>


class GlobSkel {
public:
    /* Functions */
    void init(std::shared_ptr<rclcpp::Node> node);
    void update_skel(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &pts, geometry_msgs::msg::TransformStamped transform);
    void refine_skel();

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_skeleton;
    pcl::PointCloud<pcl::PointXYZ>::Ptr debug_cloud;

    /* Utils */



private:
    /* Params */
    double tolerance = 1.0;
    int seg_count;

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tf_cloud;
    int skel_size;
    int new_pts;
    Eigen::MatrixXd branch_points;
    Eigen::MatrixXi skel_;

    /* Utils */
    std::unique_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // kdtree for skeleton refinement
    pcl::VoxelGrid<pcl::PointXYZ> vgf;

};
#endif
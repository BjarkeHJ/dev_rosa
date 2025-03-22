#ifndef __GLOBAL_SKEL__
#define __GLOBAL_SKEL__

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>


class GlobSkel {
public:
    /* Functions */
    void init(std::shared_ptr<rclcpp::Node> node);
    void update_skel(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &pts, geometry_msgs::msg::TransformStamped transform);
    void refine_skel();

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_skeleton;

    /* Utils */


private:
    /* Params */
    double tolerance = 1.0;
    int seg_count;

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tf_cloud;
    int skel_size;
    std::vector<std::vector<int>> segments;

    /* Utils */
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree;

};
#endif
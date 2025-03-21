#ifndef __GLOBAL_SKEL__
#define __GLOBAL_SKEL__

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <pcl/common/common.h>
#include <pcl/octree/octree_search.h>


class GlobSkel {
public:
    /* Functions */
    void init(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<tf2_ros::Buffer> &tfbuf);
    void update_skel();

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_rosa_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_skeleton;

    /* Utils */


private:
    /* Params */
    int skel_KNN;
    double tolerance = 1.0;

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tf_cloud; 

    /* Utils */
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree;

};
#endif
#include <iostream>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>

#ifndef __ROSA_MAIN__
#define __ROSA_MAIN__

struct SkeletonDecomposition 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;

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
    void normal_estimation();
    void normalize();
    void drosa();
    void dcrosa();

    /* Params */
    int ne_KNN; // K normal estimation neighbours
    
    /* Data */
    int pcd_size_;
    double norm_scale;
    Eigen::Vector4d centroid;

    /* Utils */
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne; // Normal estimation 


};

#endif //ROSA_MAIN
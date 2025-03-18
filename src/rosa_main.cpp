#include "rosa_main.hpp"

void RosaMain::init(std::shared_ptr<rclcpp::Node> node) {
    /* Get launch parameters */
    node->declare_parameter<int>("rosa_main/normal_est_KNN", 10);
    ne_KNN = node->get_parameter("rosa_main/normal_est_KNN").as_int();
    RCLCPP_INFO(node->get_logger(), "Test: %d", ne_KNN);

    /* Initialize data structures */
    SSD.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void RosaMain::main() {
    if (SSD.pts_->empty()) return;
    
    pcd_size_ = SSD.pts_->points.size();
    normalize();
}

void RosaMain::normalize() {
    /* Normalization */
    pcl::PointXYZ min, max;
    pcl::getMinMax3D(*SSD.pts_, min, max);
    double x_scale, y_scale, z_scale;
    x_scale = max.x - min.x;
    y_scale = max.y - min.y;
    z_scale = max.z - min.z;
    norm_scale = std::max(x_scale, std::max(y_scale, z_scale));
    pcl::compute3DCentroid(*SSD.pts_, centroid);

    for (int i=0; i<pcd_size_; i++) {
        SSD.pts_->points[i].x = (SSD.pts_->points[i].x - centroid(0)) / norm_scale;
        SSD.pts_->points[i].y = (SSD.pts_->points[i].y - centroid(1)) / norm_scale;
        SSD.pts_->points[i].z = (SSD.pts_->points[i].z - centroid(2)) / norm_scale;
    }

    /* Normal Estimation */
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setInputCloud(SSD.pts_);
    ne.setSearchMethod(tree);
    ne.setKSearch(ne_KNN);
}






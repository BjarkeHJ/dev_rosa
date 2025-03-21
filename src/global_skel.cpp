#include "global_skel.hpp"

void GlobSkel::init(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<tf2_ros::Buffer> &tfbuf) {
    tf_buffer = tfbuf;
    global_skeleton.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_rosa_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    temp_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    tf_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void GlobSkel::update_skel() {
    if (local_rosa_pts->empty()) return;

    temp_cloud->clear();
    
    geometry_msgs::msg::TransformStamped curr_tf;
    curr_tf = tf_buffer->lookupTransform("World", "lidar_frame", tf2::TimePointZero);
    
    for (const auto p : local_rosa_pts->points) {
        pcl::PointXYZ pt;
        pt.x = p.x + curr_tf.transform.translation.x;
        pt.y = p.y + curr_tf.transform.translation.y;
        pt.z = p.z + curr_tf.transform.translation.z;
        temp_cloud->points.push_back(pt);
    }
    tf_cloud = temp_cloud;
    temp_cloud->clear();

    octree->setInputCloud(global_skeleton);
    octree->addPointsFromInputCloud();

    for (const auto &pt : local_rosa_pts->points) {
        std::vector<int> pt_idx_search;
        std::vector<float> pt_sq_dist;

        if (!octree->radiusSearch(pt, tolerance, pt_idx_search, pt_sq_dist)) {
            temp_cloud->points.push_back(pt);
            octree->addPointToCloud(pt, global_skeleton);
        }
    }
    *global_skeleton += *temp_cloud;
    std::cout << "Skeleton Size: " << global_skeleton->points.size() << std::endl;
}

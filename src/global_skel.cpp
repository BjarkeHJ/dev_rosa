#include "global_skel.hpp"

void GlobSkel::init(std::shared_ptr<rclcpp::Node> node) {
    node->declare_parameter<double>("tolerance", 1.0);
    tolerance = node->get_parameter("tolerance").as_double();

    global_skeleton.reset(new pcl::PointCloud<pcl::PointXYZ>);
    temp_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    tf_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(tolerance));

    seg_count = 0;
}

void GlobSkel::update_skel(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &pts, geometry_msgs::msg::TransformStamped transform) {
    if (!pts || pts->empty()) return;

    temp_cloud->clear();

    octree->setInputCloud(global_skeleton);
    octree->addPointsFromInputCloud();

    for (const auto &pt : pts->points) {
        std::vector<int> pt_idx_search;
        std::vector<float> pt_sq_dist;

        pcl::PointXYZ pt_t;
        pt_t.x = pt.x + transform.transform.translation.x;
        pt_t.y = pt.y + transform.transform.translation.y;
        pt_t.z = pt.z + transform.transform.translation.z;

        if (!octree->radiusSearch(pt_t, tolerance, pt_idx_search, pt_sq_dist)) {
            temp_cloud->points.push_back(pt_t);
            octree->addPointToCloud(pt_t, global_skeleton);
        }
    }
    *global_skeleton += *temp_cloud;
    skel_size = global_skeleton->points.size();
    
    refine_skel();

}

void GlobSkel::refine_skel() {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr refine_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    refine_tree->setInputCloud(global_skeleton);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(2*tolerance);
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(200);
    ec.setSearchMethod(refine_tree);
    ec.setInputCloud(global_skeleton);

    std::vector<pcl::PointIndices> cluster_indxs;
    ec.extract(cluster_indxs);
    
    std::cout << "Custer indxs size: " << cluster_indxs.size() << std::endl;

    // For each point 
        // check position in terms of general direction of previous point
            // if distance from search point < 2xtolerance
                // consider inlier
            // if outlier of any segment
                // increment segment counter
                    // add first point as segment seed

        // for each segment 
            // grow branch

    // Refine on last X points
    




    // Notes
    // pcl::regionGrowing for straight line identification 
    // pcl::SACSegmentation for straight line

}

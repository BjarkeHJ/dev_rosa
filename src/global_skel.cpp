#include "global_skel.hpp"

void GlobSkel::init(std::shared_ptr<rclcpp::Node> node) {
    node->declare_parameter<double>("tolerance", 1.0);
    tolerance = node->get_parameter("tolerance").as_double();

    global_skeleton.reset(new pcl::PointCloud<pcl::PointXYZ>);
    temp_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    tf_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(tolerance));

    debug_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    seg_count = 0;
    branch_points.resize(0,3);
    skel_.resize(0,1);
}

void GlobSkel::update_skel(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &pts, geometry_msgs::msg::TransformStamped transform) {
    if (!pts || pts->empty()) return;

    temp_cloud->clear();

    octree->setInputCloud(global_skeleton);
    octree->addPointsFromInputCloud();
    
    new_pts = 0;
    for (const auto &pt : pts->points) {
        std::vector<int> pt_idx_search;
        std::vector<float> pt_sq_dist;

        pcl::PointXYZ pt_t;
        pt_t.x = pt.x + transform.transform.translation.x;
        pt_t.y = pt.y + transform.transform.translation.y;
        pt_t.z = pt.z + transform.transform.translation.z;

        if (!octree->radiusSearch(pt_t, tolerance, pt_idx_search, pt_sq_dist)) {
            new_pts++;
            octree->addPointToCloud(pt_t, global_skeleton);
        }
    }

    refine_skel();
    std::cout << "number of branch point found: " << branch_points.rows() << std::endl;
}

void GlobSkel::refine_skel() {
    vgf.setInputCloud(global_skeleton);
    vgf.setLeafSize(tolerance, tolerance, tolerance);
    vgf.filter(*global_skeleton);

    skel_size = global_skeleton->points.size();
    skel_.conservativeResize(skel_size, 1); // increment corresponding to new pts
    skel_.bottomRows(new_pts).setConstant(-1); // Set new points uninitialized

    std::vector<int> outlier_idxs;

    kdtree.setInputCloud(global_skeleton);
    for (int i=0; i<new_pts; i++) {
        int curr_idx = skel_size - new_pts + i;
        if (skel_(curr_idx, 0) == -1) {
            pcl::PointXYZ pt = global_skeleton->points[curr_idx];
            std::vector<int> pt_idxs;
            std::vector<float> pt_sq_dist;

            if (kdtree.radiusSearch(pt, 1.5*tolerance, pt_idxs, pt_sq_dist) > 0) {
                if ((int)pt_idxs.size() > 2) {
                    skel_(curr_idx,0) = 1; // Mark as branch joing
                    branch_points.conservativeResize(branch_points.rows() + 1, branch_points.cols());
                    branch_points.bottomRows(1) = Eigen::RowVector3d(pt.x, pt.y, pt.z);

                    debug_cloud->points.push_back(pt); // For visualizing branch points
                }
                else { 
                    skel_(curr_idx,0) = 0; // Mark as part of branch
                }
            }
            else {
                outlier_idxs.push_back(curr_idx);
            }
        }
        else continue;
    } 
    std::sort(outlier_idxs.rbegin(), outlier_idxs.rend());
    for (int idx : outlier_idxs) {
        global_skeleton->points.erase(global_skeleton->points.begin() + idx);
        skel_.conservativeResize(skel_.rows() - 1, 1);
    }
}

// TRY SVD FOR DETECTING PRINCIPAL DIRECTION INSTEAD!!

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
// Get normal symmetry normal vectors for remaining rosa points transferred to here...
    // use these to do directional clustering
// pcl::regionGrowing for straight line identification 
// pcl::SACSegmentation for straight line
    
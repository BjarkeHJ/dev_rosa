#ifndef __ROSA_MAIN__
#define __ROSA_MAIN__

#include <iostream>
#include <chrono>
#include <deque>
#include <stack>
#include <algorithm>
#include <map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl_ros/transforms.hpp>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Core>

#include "Extra_Del.hpp"

struct Vector3dCompare // lexicographic ordering: return true if v1 is ordered BEFORE v2...
{
    bool operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
        if (v1(0) != v2(0)) return v1(0) < v2(0); // Return if x components differ (False if x1 > x2)
        if (v1(1) != v2(1)) return v1(1) < v2(1); // Only if x1 = x2 (False if y1 > y2)
        return v1(2) < v2(2); // Only if y1 = y2
    }
};

struct Vector3iCompare // lexicographic ordering: return true if v1 is ordered BEFORE v2...
{
    bool operator()(const Eigen::Vector3i& v1, const Eigen::Vector3i& v2) const {
        if (v1(0) != v2(0)) return v1(0) < v2(0); // Return if x components differ (False if x1 > x2)
        if (v1(1) != v2(1)) return v1(1) < v2(1); // Only if x1 = x2 (False if y1 > y2)
        return v1(2) < v2(2); // Only if y1 = y2
    }
};


struct SkeletonDecomposition 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals;
    Eigen::MatrixXd pts_matrix;
    Eigen::MatrixXd nrs_matrix;

    std::vector<std::vector<int>> neighs; // For each point x in pts_ store the indices of the neighbouring point int neighs[x].
    std::vector<std::vector<int>> neighs_new;
    std::vector<std::vector<int>> neighs_surf;

    Eigen::MatrixXd skelver; // Current skeleton vertices
    Eigen::MatrixXd corresp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr rosa_pts; // Rosa pts for global skeleton increment 
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_skeleton;

    Eigen::MatrixXd global_vertices; // Global skeleton vertices
    Eigen::MatrixXd global_edges; // Global skeleton edges
    Eigen::MatrixXi global_adj; // Global skeleton adjacency matrix


};

class RosaMain {
public:
    /* Functions */
    void init(std::shared_ptr<rclcpp::Node> node);
    void main();

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr debug_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr debug_cloud_2;

    /* Utils */
    SkeletonDecomposition SSD;
    geometry_msgs::msg::TransformStamped transform;

private:
    /* Functions */
    void distance_filter();
    void normalize();
    void mahanalobis_mat(double &radius_r);
    double pt_mahalanobis_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double &range_r);
    void drosa();
    void dcrosa();
    void vertex_sampling();
    void vertex_recenter();
    void restore_scale();
    void incremental_graph();
    void lineextraction();

    void rosa_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals);
    Eigen::Matrix3d create_orthonormal_frame(Eigen::Vector3d &v);
    Eigen::MatrixXd compute_active_samples(int &idx, Eigen::Vector3d &p_cut, Eigen::Vector3d &v_cut);
    Eigen::Vector3d compute_symmetrynormal(Eigen::MatrixXd& local_normals);
    double symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals);
    Eigen::Vector3d symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w);
    Eigen::Vector3d closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V);
    int argmax_eigen(Eigen::MatrixXd &x);
    
    /* Params */
    double pts_dist_lim; // For lidar point distance filtering
    int ne_KNN; // K normal estimation neighbours
    double radius_neigh; // Radius for mahanalobis neighbour search
    double th_mah; // Mahanalobis distance threshold
    int nMax; // Maximum number of points in each point cloud
    int nMin; // Minimum ...
    double leaf_size_ds; // Set dynamically... 
    int k_KNN; // Number of neighbours in surface-neighbour search (drosa / dcrosa)
    int drosa_iter; // Number of iteration in drosa
    int dcrosa_iter; // Number of iteration in dcrosa
    double delta; // Plane slice thickness -- Will be set equal to leaf_size_ds once determined
    double sample_radius; // Sample radius for line extraction
    double alpha_recenter; // rosa recentering...
    double tolerance; // Tolerance for incremental skeleton 
    
    /* Data */
    int pcd_size_;
    double norm_scale;
    Eigen::Vector4d centroid;
    Eigen::MatrixXd pset; //point set
    Eigen::MatrixXd vset; //symm vector set
    Eigen::MatrixXd vvar;  //symm vector variance
    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud;
    Eigen::MatrixXi adj_before_collapse;
    
    /* Utils */
    std::unique_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> global_octree;

};


#endif //ROSA_MAIN
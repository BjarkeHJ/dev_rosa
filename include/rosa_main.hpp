#ifndef __ROSA_MAIN__
#define __ROSA_MAIN__

#include <iostream>
#include <chrono>
#include <deque>
#include <stack>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl_ros/transforms.hpp>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>

#include "Extra_Del.hpp"

class DataWrapper {
private:
    double* data;
    int npoints;
    const static int ndim = 3; 
        
public: 
    void factory(double* data, int npoints ) {
        this->data = data;
        this->npoints = npoints;
    }
    /** 
     *  Data retrieval function
     *  @param a address over npoints
     *  @param b address over the dimensions
     */
    inline double operator()(int a, int b) {
        assert( a < npoints );
        assert( b < ndim );
        return data[ a + npoints*b ];
    }
    // retrieve a single point at offset a, in a vector (preallocated structure)
    inline void operator()(int a, std::vector<double>& p){
        assert( a < npoints );
        assert( (int)p.size() == ndim );
        p[0] = data[ a + 0*npoints ];
        p[1] = data[ a + 1*npoints ];
        p[2] = data[ a + 2*npoints ];
    }
    int length(){
        return this->npoints;
    }
};

struct Vector3dCompare // lexicographic ordering: return true if v1 is ordered BEFORE v2...
{
    bool operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
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
    double *datas;

    std::vector<std::vector<int>> neighs; // For each point x in pts_ store the indices of the neighbouring point int neighs[x].
    std::vector<std::vector<int>> neighs_new;
    std::vector<std::vector<int>> neighs_surf;

    Eigen::MatrixXd skelver;
    Eigen::MatrixXd corresp;
    Eigen::MatrixXi skeladj;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges;
    Eigen::MatrixXi degrees;
    std::deque<int> joint;
    std::vector<std::list<int>> graph;
    std::vector<std::vector<int>> branches;
    std::vector<bool> visited;

    pcl::PointCloud<pcl::PointXYZ>::Ptr rosa_pts;

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

private:
    /* Functions */
    void normalize();
    void mahanalobis_mat(double &radius_r);
    double pt_mahalanobis_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double &range_r);
    void drosa();
    void dcrosa();
    void lineextract();
    void recenter();
    void restore_scale();
    void refine_points();

    // void graph_decomposition();
    // void inner_decomposition();

    void rosa_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals);
    Eigen::Matrix3d create_orthonormal_frame(Eigen::Vector3d &v);
    Eigen::MatrixXd compute_active_samples(int &idx, Eigen::Vector3d &p_cut, Eigen::Vector3d &v_cut);
    void pcloud_isoncut(Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut, std::vector<int>& isoncut, double*& datas, int& size);
    void distance_query(DataWrapper& data, const std::vector<double>& Pp, const std::vector<double>& Np, double delta, std::vector<int>& isoncut);
    Eigen::Vector3d compute_symmetrynormal(Eigen::MatrixXd& local_normals);
    double symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals);
    Eigen::Vector3d symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w);
    Eigen::Vector3d closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V);
    int argmax_eigen(Eigen::MatrixXd &x);

    void dfs(int &v);
    bool ocr_node(int &n, std::list<int> &candidates);
    std::vector<std::vector<int>> divide_branch(std::vector<int> &input_branch);


    /* Params */
    int ne_KNN; // K normal estimation neighbours
    double radius_neigh; // Radius for mahanalobis neighbour search
    double th_mah; // Mahanalobis distance threshold
    int nMax; // Maximum number of points in each point cloud
    double leaf_size_ds;
    int k_KNN; // Number of neighbours in surface-neighbour search (drosa / dcrosa)
    int drosa_iter; // Number of iteration in drosa
    int dcrosa_iter; // Number of iteration in dcrosa
    double delta; // Plane slice thickness
    double sample_radius; // Sample radius for line extraction
    double alpha_recenter; // rosa recentering...
    double angle_upper; // Upper angle bound for inner decomp...
    double length_upper; // upper length bound for inner decomp...

    /* Data */
    int pcd_size_;
    double norm_scale;
    Eigen::Vector4d centroid;
    Eigen::MatrixXd pset; //Point set
    Eigen::MatrixXd vset; //Vector set
    Eigen::MatrixXd vvar;  //Vector variance
    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr skeleton_ver_cloud;
    Eigen::MatrixXi adj_before_collapse;

    /* Utils */
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne; // Normal estimation
    pcl::VoxelGrid<pcl::PointNormal> vgf; // Voxel Grid Filter for downsamplings
    pcl::KdTreeFLANN<pcl::PointXYZ> surf_kdtree;
    pcl::KdTreeFLANN<pcl::PointXYZ> rosa_tree;
    pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;

};

#endif //ROSA_MAIN
#include "rosa_main.hpp"

void RosaMain::init(std::shared_ptr<rclcpp::Node> node) {
    /* Get launch parameters */
    node->declare_parameter<int>("normal_est_KNN", 10);
    node->declare_parameter<double>("neighbour_radius", 0.1);
    ne_KNN = node->get_parameter("normal_est_KNN").as_int();
    radius_neigh = node->get_parameter("neighbour_radius").as_double();

    /* Initialize data structures */
    SSD.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.normals_.reset(new pcl::PointCloud<pcl::Normal>);
    SSD.cloud_w_normals.reset(new pcl::PointCloud<pcl::PointNormal>);

    th_mah = 0.1 * radius_neigh;
}

void RosaMain::main() {
    if (SSD.pts_->empty()) return;

    pcd_size_ = SSD.pts_->points.size();
    normalize();
    mahanalobis_mat(radius_neigh);
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
    SSD.normals_->clear();
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setInputCloud(SSD.pts_);
    ne.setSearchMethod(tree);
    ne.setKSearch(ne_KNN);
    ne.compute(*SSD.normals_); 

    pcl::concatenateFields(*SSD.pts_, *SSD.normals_, *SSD.cloud_w_normals);

    /* Voxel Grid Downsample */
    double leaf_ds_size = norm_scale / 10000;
    vgf.setInputCloud(SSD.cloud_w_normals);
    vgf.setLeafSize(leaf_ds_size, leaf_ds_size, leaf_ds_size);
    vgf.filter(*SSD.cloud_w_normals);

    pcd_size_ = SSD.cloud_w_normals->points.size();

    SSD.pts_->clear();
    SSD.normals_->clear();
    SSD.pts_matrix.resize(pcd_size_, 3);
    SSD.nrs_matrix.resize(pcd_size_, 3);
    pcl::PointXYZ pt;
    pcl::Normal nrm;

    for (int i=0; i<pcd_size_; i++) {
        pt.x = SSD.cloud_w_normals->points[i].x;
        pt.y = SSD.cloud_w_normals->points[i].y;
        pt.z = SSD.cloud_w_normals->points[i].z;
        nrm.normal_x = SSD.cloud_w_normals->points[i].normal_x;
        nrm.normal_y = SSD.cloud_w_normals->points[i].normal_y;
        nrm.normal_z = SSD.cloud_w_normals->points[i].normal_z;
        SSD.pts_->points.push_back(pt);
        SSD.normals_->points.push_back(nrm);
        SSD.pts_matrix(i,0) = pt.x;
        SSD.pts_matrix(i,1) = pt.y;
        SSD.pts_matrix(i,2) = pt.z;
        SSD.nrs_matrix(i,0) = nrm.normal_x;
        SSD.nrs_matrix(i,1) = nrm.normal_y;
        SSD.nrs_matrix(i,2) = nrm.normal_z;
    }
}


void RosaMain::mahanalobis_mat(double &radius_r) {
    /* Neighbour searhc based on correlation between neighbouring normal vectors */
    SSD.neighs.clear();
    SSD.neighs.resize(pcd_size_);

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(SSD.pts_);

    pcl::PointXYZ search_pt, p1, p2;
    pcl::Normal v1, v2;
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    double w1, w2, w;
    std::vector<std::vector<int>> pt_neighs_idx;

    for (int i=0; i<pcd_size_; i++) {
        std::vector<int>().swap(indxs); // efficiently clears and reallocates memory (since size varies in neighbour computations)
        std::vector<float>().swap(radius_squared_distance);
        p1 = SSD.pts_->points[i]; // current search point
        v1 = SSD.normals_->points[i];
        tree.radiusSearch(p1, radius_r, indxs, radius_squared_distance); // Radius search
        std::vector<int> temp_neighs;

        for (int j=0; j<(int)indxs.size(); j++) {
            p2 = SSD.pts_->points[indxs[j]];
            v2 = SSD.normals_->points[indxs[j]];
            w1 = pt_mahalanobis_metric(p1, v1, p2, v2, radius_r);
            w2 = pt_mahalanobis_metric(p2, v2, p1, v1, radius_r);
            w = std::min(w1, w2);

            if (w > th_mah) {
                temp_neighs.push_back(indxs[j]);
            }
        }
        std::cout << temp_neighs.size() << std::endl;
        SSD.neighs[i] = temp_neighs;
    }
}

double RosaMain::pt_mahalanobis_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double &range_r) {
    double Fs = 2.0;
    double k = 0.0;
    double dist, vec_dot, w;
    Eigen::Vector3d p1_, p2_, v1_, v2_;

    p1_ << p1.x, p1.y, p1.z;
    p2_ << p2.x, p2.y, p2.z;
    v1_ << v1.normal_x, v1.normal_y, v1.normal_z;
    v2_ << v2.normal_x, v2.normal_y, v2.normal_z;

    dist = (p1_ - p2_ + Fs*((p1_ - p2_).dot(v1_))*v1_).norm();
    dist = dist/range_r;

    if (dist <= 1) {
        k = 2*pow(dist, 3) - 3*pow(dist, 2) + 1;
    }

    vec_dot = v1_.dot(v2_);
    w = k*pow(std::max(0.0, vec_dot), 2);
    return w;
}

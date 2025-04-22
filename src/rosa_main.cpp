#include "rosa_main.hpp"

void RosaMain::init(std::shared_ptr<rclcpp::Node> node) {
    /* Get launch parameters */
    node->declare_parameter<double>("max_lidar_dist", 20);
    node->declare_parameter<int>("normal_est_KNN", 10);
    node->declare_parameter<double>("neighbour_radius", 0.1);
    node->declare_parameter<int>("max_pts", 1000);
    node->declare_parameter<int>("min_pts", 50);
    node->declare_parameter<int>("neighbour_KNN", 6);
    node->declare_parameter<int>("drosa_iter", 1);
    node->declare_parameter<int>("dcrosa_iter", 1);
    node->declare_parameter<double>("sample_radius", 0.05);
    node->declare_parameter<double>("alpha", 0.3);
    node->declare_parameter<double>("tolerance", 1.0);
    node->declare_parameter<double>("kf_dist_th", 1.0);
    node->declare_parameter<double>("kf_conf_th", 1.0);
    node->declare_parameter<double>("kf_process_noise", 1.0);
    node->declare_parameter<double>("kf_meas_noise", 1.0);

    pts_dist_lim = node->get_parameter("max_lidar_dist").as_double();
    ne_KNN = node->get_parameter("normal_est_KNN").as_int();
    radius_neigh = node->get_parameter("neighbour_radius").as_double();
    nMax = node->get_parameter("max_pts").as_int();
    nMin = node->get_parameter("min_pts").as_int();
    k_KNN = node->get_parameter("neighbour_KNN").as_int();
    drosa_iter = node->get_parameter("drosa_iter").as_int();
    dcrosa_iter = node->get_parameter("dcrosa_iter").as_int();
    sample_radius = node->get_parameter("sample_radius").as_double();
    alpha_recenter = node->get_parameter("alpha").as_double();
    tolerance = node->get_parameter("tolerance").as_double();
    kf_dist_th = node->get_parameter("kf_dist_th").as_double();
    kf_conf_th = node->get_parameter("kf_conf_th").as_double();
    kf_pn = node->get_parameter("kf_process_noise").as_double();
    kf_mn = node->get_parameter("kf_meas_noise").as_double();

    /* Initialize data structures */
    global_octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(tolerance));
    SSD.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.normals_.reset(new pcl::PointCloud<pcl::Normal>);
    SSD.cloud_w_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
    pset_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.rosa_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.ver_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.global_skeleton.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.global_vertices.resize(0,3);

    debug_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    debug_cloud_2.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pts_dist_filt.reset(new pcl::PointCloud<pcl::PointXYZ>);

    th_mah = 0.1 * radius_neigh;
}

void RosaMain::main() {
    auto start = std::chrono::high_resolution_clock::now();
    distance_filter();

    pcd_size_ = SSD.pts_->points.size();
    if (pcd_size_ == 0) return; // No points within range...

    std::cout << "Distance to structure: " << closest_point << std::endl;

    normalize();
    if (pcd_size_ < nMin) {
        std::cout << "Point Cloud size below nMin... Skipping..." << std::endl; 
        return; // Too few points to reliably compute ROSA pts
    }

    debug_cloud_2 = scale_transform_debugger(SSD.pts_);
    pts_tf = scale_transform_debugger_matmat(SSD.pts_matrix);

    /* Main Rosa Algorithm */
    mahanalobis_mat(radius_neigh);
    drosa();
    dcrosa();

    // Extract points and transform them...
    // debug_cloud = scale_transform_debugger(pset);


    // std::cout << "Before sampling..." << std::endl;
    vertex_sampling();
    // vertex_sampling_kmeans();

    local_lineextract();
    // std::cout << "Before recenter..." << std::endl;


    vertex_recenter();
    // std::cout << "Before restoring..." << std::endl;

    restore_scale();

    /* Global Skeleton Extraction */
    // kf_skeleton_incr();
    // graph_adj();


    // global_lineextraction();
    // mst();


    // graph_decomp();
    // vertex_merge();
    // branch_extract();

    
    // prune_branches();
    // update_skeleton();

    // for (int i=0; i<(int)SSD.gadj.rows(); ++i) {
        //     for (int j=0; j<(int)SSD.gadj.cols(); ++j) {
            //         std::cout << SSD.gadj(i,j) << " ";
            //     }
            //     std::cout << "" << std::endl;
            // }
    
    // TODO: 
    // Properly segment branches based on directional similarity
    // Store branches in std::map<int, Eigen::MatrixXd> 
    // For each branch compute the main direction (PCA)
    // Recenter points in each branch by projecting points onto the direction as:
    // proj = centroid + main_dir * (point - centroid).dot(main_dir)
    
    // global_lineextraction();
    // When extracting branches: When visualizing recolor the indices of each branch differently...
    
    
    
    // Merge vertices based on areas density (mean) - Thinking skeleton extraction of nacelle and house in general... (???)
    // Do joint handling 
    
    
    
    // Extract current active global vertices
    
    
    
    // std::cout << "Number of Branches: " << SSD.branches.size() << std::endl;
    // std::cout << "Global Skeleton Size: " << SSD.global_skeleton->points.size() << std::endl;
    // std::cout << "Global Adjacency Matrix Size: " << SSD.gadj.rows() << std::endl;
    // std::cout << "Number of joints: " << SSD.joint_ids.size() << std::endl;

    // Generate viewpoints in XY-Plane on either side of the skeleton (three sides if endpoint)
    // When visiting a viewpoint veryfy its validity (???)
    // Mark viewpoints / vertices visited (for vertices 1/2 when one side is visited) (???)

    // incremental_graph();
    // global_lineextraction();

    // debug_cloud = SSD.global_skeleton;

    debug_cloud->points.clear();
    for (int i=0; i<(int)SSD.skelver_scaled.rows(); ++i) {
        pcl::PointXYZ pt;
        pt.x = SSD.skelver_scaled(i,0);
        pt.y = SSD.skelver_scaled(i,1);
        pt.z = SSD.skelver_scaled(i,2);
        debug_cloud->points.push_back(pt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}

void RosaMain::distance_filter() {
    pts_dist_filt->points.clear();
    pcl::PassThrough<pcl::PointXYZ> ptf;
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ptf.setInputCloud(SSD.pts_);
    ptf.setFilterFieldName("x");
    ptf.setFilterLimits(-pts_dist_lim, pts_dist_lim);
    ptf.filter(*temp_cloud);  

    ptf.setInputCloud(temp_cloud);
    ptf.setFilterFieldName("y");
    ptf.setFilterLimits(-pts_dist_lim, pts_dist_lim);
    ptf.filter(*pts_dist_filt);
    pcl::copyPointCloud(*pts_dist_filt, *SSD.pts_);

    // Compute the closest distance to the structure...
    if (pts_dist_filt->points.empty()) return;
    pcl::KdTreeFLANN<pcl::PointXYZ> cp_tree;
    cp_tree.setInputCloud(pts_dist_filt);
    pcl::PointXYZ pt(0.0, 0.0, 0.0);
    int K = 10;
    std::vector<int> pt_ids;
    std::vector<float> pt_dists;
    if (cp_tree.nearestKSearch(pt, K, pt_ids, pt_dists) > 0) {
        closest_point = 0.0;
        for (int i=0; i<(int)pt_dists.size(); i++) {
            closest_point += std::sqrt(pt_dists[i]);
        }
        closest_point /= static_cast<double>(K);
    }
}

void RosaMain::normalize() {
    /* Normalization */
    pcl::PointXYZ min, max;
    pcl::getMinMax3D(*SSD.pts_, min, max);
    // double x_scale, y_scale, z_scale;
    maxx = max.x - min.x;
    maxy = max.y - min.y;
    maxz = max.z - min.z;
    norm_scale = std::max(maxx, std::max(maxy, maxz));

    pcl::compute3DCentroid(*SSD.pts_, centroid);

    for (int i=0; i<pcd_size_; i++) {
        SSD.pts_->points[i].x = (SSD.pts_->points[i].x - centroid(0)) / norm_scale;
        SSD.pts_->points[i].y = (SSD.pts_->points[i].y - centroid(1)) / norm_scale;
        SSD.pts_->points[i].z = (SSD.pts_->points[i].z - centroid(2)) / norm_scale;
    }
    
    SSD.normals_->clear();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ne_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(SSD.pts_);
    ne.setSearchMethod(ne_tree);
    // ne.setKSearch(ne_KNN);
    ne.setRadiusSearch(0.05);
    ne.compute(*SSD.normals_); 

    pcl::concatenateFields(*SSD.pts_, *SSD.normals_, *SSD.cloud_w_normals);

    /* Dynamic Voxel Grid Downsampling */
    if (pcd_size_ > nMax) {
        pcl::VoxelGrid<pcl::PointNormal> vgf;
        leaf_size_ds = 0.001;
        while (pcd_size_ > nMax) {
            vgf.setInputCloud(SSD.cloud_w_normals);
            vgf.setLeafSize(leaf_size_ds, leaf_size_ds, leaf_size_ds);
            vgf.filter(*SSD.cloud_w_normals);
            pcd_size_ = SSD.cloud_w_normals->points.size();
            if (pcd_size_ <= nMax) break;
            leaf_size_ds += 0.001;
        }
    }

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
        
        Eigen::Vector3d pt_vec(pt.x, pt.y, pt.z);
        Eigen::Vector3d nrm_vec(SSD.cloud_w_normals->points[i].normal_x,
                                SSD.cloud_w_normals->points[i].normal_y,
                                SSD.cloud_w_normals->points[i].normal_z);
        Eigen::Vector3d lidar_pos = (-centroid.head<3>().transpose()) / norm_scale;
        Eigen::Vector3d to_sensor = lidar_pos-pt_vec;

        if (nrm_vec.dot(to_sensor) < 0) {
            nrm_vec = -nrm_vec;
        }

        nrm.normal_x = nrm_vec(0);
        nrm.normal_y = nrm_vec(1);
        nrm.normal_z = nrm_vec(2);

        // nrm.normal_x = -SSD.cloud_w_normals->points[i].normal_x;
        // nrm.normal_y = -SSD.cloud_w_normals->points[i].normal_y;
        // nrm.normal_z = -SSD.cloud_w_normals->points[i].normal_z;
        SSD.pts_->points.push_back(pt);
        SSD.normals_->points.push_back(nrm);
        SSD.pts_matrix(i,0) = pt.x;
        SSD.pts_matrix(i,1) = pt.y;
        SSD.pts_matrix(i,2) = pt.z;
        SSD.nrs_matrix(i,0) = nrm.normal_x;
        SSD.nrs_matrix(i,1) = nrm.normal_y;
        SSD.nrs_matrix(i,2) = nrm.normal_z;
    }

    th_mah = 0.1*radius_neigh; // Threshold for similarity neighbour extraction (original 0.1 * radius_neigh)
    delta = leaf_size_ds; // original: Plane slice thickness kept equal to the voxel leaf size
}

void RosaMain::mahanalobis_mat(double &radius_r) {
    /* Neighbour searhc based on correlation between neighbouring normal vectors */
    SSD.neighs.clear();
    SSD.neighs.resize(pcd_size_);

    pcl::KdTreeFLANN<pcl::PointXYZ> maha_tree;
    maha_tree.setInputCloud(SSD.pts_);
    pcl::PointXYZ search_pt, p1, p2;
    pcl::Normal v1, v2;
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    double w1, w2, w;
    std::vector<std::vector<int>> pt_neighs_idx;

    radius_r = 5 * leaf_size_ds;
    std::cout << "Maha neigh distance: " << radius_r << std::endl;

    for (int i=0; i<pcd_size_; i++) {
        std::vector<int>().swap(indxs); // efficiently clears and reallocates memory (since size varies in neighbour computations)
        std::vector<float>().swap(radius_squared_distance);
        p1 = SSD.pts_->points[i]; // current search point
        v1 = SSD.normals_->points[i];
        maha_tree.radiusSearch(p1, radius_r, indxs, radius_squared_distance); // Radius search
        std::vector<int> temp_neighs;

        for (int j=0; j<(int)indxs.size(); j++) {
            p2 = SSD.pts_->points[indxs[j]];
            v2 = SSD.normals_->points[indxs[j]];
            w1 = pt_mahalanobis_metric(p1, v1, p2, v2, radius_r);
            w2 = pt_mahalanobis_metric(p2, v2, p1, v1, radius_r);
            w = std::min(w1, w2);

            // Exclude all bad and close-to bad nbs...
            if (w > th_mah) {
                temp_neighs.push_back(indxs[j]);
            }
        }

        // std::cout << "Maha Neighs: " << temp_neighs.size() << std::endl;

        // Normal aware neighbors
        SSD.neighs[i] = temp_neighs;
    }
}

double RosaMain::pt_mahalanobis_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double &range_r) {
    double Fs = 10.0;
    double k = 0.0;
    double dist, vec_dot, w;
    Eigen::Vector3d p1_, p2_, v1_, v2_;

    p1_ << p1.x, p1.y, p1.z;
    p2_ << p2.x, p2.y, p2.z;
    v1_ << v1.normal_x, v1.normal_y, v1.normal_z;
    v2_ << v2.normal_x, v2.normal_y, v2.normal_z;

    // the displacement vector + the projection of the displacement vector onto the search point normal vector
    // If the displacement vector is perpendicular with the normal vector (projection = 0) the two points both lie in the plane given by the normal vector
    // If that is the case, the contribution to the distance metric is not increased
    // Else the metric is increased... 
    dist = (p1_ - p2_ + Fs*((p1_ - p2_).dot(v1_))*v1_).norm();
    dist = dist/range_r;

    if (dist <= 1) {
        k = 2*pow(dist, 3) - 3*pow(dist, 2) + 1;
    }

    // Projection of v1 onto v2
    vec_dot = v1_.dot(v2_);
    // max(0, vec_dot) to not include antiparallel normal vectors
    w = k*pow(std::max(0.0, vec_dot), 2);
    return w;
}

void RosaMain::drosa() {
    Extra_Del ed_;
    rosa_initialize(SSD.pts_, SSD.normals_);

    std::vector<std::vector<int>>().swap(SSD.neighs_surf);
    std::vector<int> temp_surf(k_KNN);
    std::vector<float> nn_squared_distance(k_KNN);
    pcl::PointXYZ search_pt_surf;

    // K Nearest neighbours search...
    pcl::KdTreeFLANN<pcl::PointXYZ> surf_tree;
    surf_tree.setInputCloud(SSD.pts_);
    for (int i=0; i<pcd_size_; i++) {
        std::vector<int>().swap(temp_surf);
        std::vector<float>().swap(nn_squared_distance);
        search_pt_surf = SSD.pts_->points[i];
        surf_tree.nearestKSearch(search_pt_surf, k_KNN, temp_surf, nn_squared_distance);
        SSD.neighs_surf.push_back(temp_surf);
    }

    Eigen::Vector3d var_p, var_v, new_v;
    Eigen::MatrixXd indxs, extract_normals;

    /* ROSA Points Orientation Calculations */
    for (int n=0; n<drosa_iter; n++) {
        Eigen::MatrixXd vnew = Eigen::MatrixXd::Zero(pcd_size_, 3);

        for (int pidx=0; pidx<pcd_size_; pidx++) {
            var_p = pset.row(pidx); // Search point for activate samples
            var_v = vset.row(pidx); // Corresponding plane normal estimate
            indxs = compute_active_samples(pidx, var_p, var_v);
            extract_normals = ed_.rows_ext_M(indxs, SSD.nrs_matrix);

            // Compute the vector that minimizes the variance of angles between local normals and itself
            vnew.row(pidx) = compute_symmetrynormal(extract_normals).transpose();
            new_v = vnew.row(pidx);

            // Compute projection variance of extracted normals on symmetry normal
            if (extract_normals.rows() > 0) {
                vvar(pidx, 0) = symmnormal_variance(new_v, extract_normals);
            }
            else {
                vvar(pidx, 0) = 0.0;
            }
        }

        vset = vnew; // Overwrite previous plane normal estimates with the updated (for iterative convergence)

        Eigen::MatrixXd offset(vvar.rows(), vvar.cols());
        offset.setOnes();
        offset = 0.00001*offset; // Ensure no division by zero

        // Weighting the variance values to suppress large variance and emphasize low-variance regions
        // Small variances have HIGH CONFIDENCE
        // Large variances are negleglible
        vvar = (vvar.cwiseAbs2().cwiseAbs2() + offset).cwiseInverse(); // 1/vvar⁴ 
        
        /* Smoothing */
        std::vector<int> surf_;
        Eigen::MatrixXi snidxs; // surface normal indices
        Eigen::MatrixXd snidxs_d;
        Eigen::MatrixXd vset_ex, vvar_ex; // Extracted vector set and corresponding vector variances
        for (int p=0; p<pcd_size_; p++) {
            std::vector<int>().swap(surf_);
            surf_ = SSD.neighs_surf[p]; // Extract neighbours of the current point
            snidxs.resize(surf_.size(), 1);
            snidxs = Eigen::Map<Eigen::MatrixXi>(surf_.data(), surf_.size(), 1);
            snidxs_d = snidxs.cast<double>(); 
            vset_ex = ed_.rows_ext_M(snidxs_d, vset);
            vvar_ex = ed_.rows_ext_M(snidxs_d, vvar);

            // Construct a weighted covariance computation based on the projection variances and 
            vset.row(p) = symmnormal_smooth(vset_ex, vvar_ex); //Overwrite the plane normal est. for next iteration...
        }
    }

    /* ROSA Points Position Calculation */
    std::vector<int> poorIdx;
    pcl::PointCloud<pcl::PointXYZ>::Ptr goodPts(new pcl::PointCloud<pcl::PointXYZ>);
    std::map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompare> goodPtsPset;
    Eigen::Vector3d var_p_p, var_v_p, center;
    Eigen::MatrixXd indxs_p, extract_pts, extract_nrs;

    for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
        var_p_p = pset.row(pIdx);
        var_v_p = vset.row(pIdx).normalized();        
        indxs_p = compute_active_samples(pIdx, var_p_p, var_v_p); // Extract active samples

        /* Update neighbours to be from the same plane slice */
        // OBS: SSD.neighs_new not used!!
        // std::vector<int> temp_neigh;
        // for (int p=0; p<(int)indxs_p.rows(); p++) {
        //     temp_neigh.push_back(indxs_p(p,0));
        // }

        // SSD.neighs_new.push_back(temp_neigh);

        extract_pts = ed_.rows_ext_M(indxs_p, SSD.pts_matrix);
        extract_nrs = ed_.rows_ext_M(indxs_p, SSD.nrs_matrix);
        center = closest_projection_point(extract_pts, extract_nrs); // Extract the intersection point of surface normals

        if (abs(center(0)) < 1 && abs(center(1)) < 1 && abs(center(2)) < 1) {
            // If the center is within the max dimension of the normalized data...
            pset.row(pIdx) = center;
            pcl::PointXYZ goodPoint;
            Eigen::Vector3d goodPointP;
            goodPoint = SSD.pts_->points[pIdx];
            goodPointP(0) = goodPoint.x;
            goodPointP(1) = goodPoint.y;
            goodPointP(2) = goodPoint.z;
            goodPts->points.push_back(goodPoint); // Pointcloud with good points
            goodPtsPset[goodPointP] = center; // adds the point to the structure if it does not already exist 
        }
        else {
            // The surface may have been too planar...
            poorIdx.push_back(pIdx);
        }
    }

    if (goodPts->points.empty()) return; // Solves crash issues where no proper points are determined...

    /* Reposition poor point to the nearest good point */
    pcl::KdTreeFLANN<pcl::PointXYZ> rosa_tree;
    rosa_tree.setInputCloud(goodPts);
    for (int pp=0; pp<(int)poorIdx.size(); pp++) {
        int pair = 1;
        pcl::PointXYZ search_pt;
        search_pt.x = SSD.pts_->points[poorIdx[pp]].x;
        search_pt.y = SSD.pts_->points[poorIdx[pp]].y;
        search_pt.z = SSD.pts_->points[poorIdx[pp]].z;
        std::vector<int> pair_id(pair);
        std::vector<float> nn_squared_distance(pair);

        rosa_tree.nearestKSearch(search_pt, pair, pair_id, nn_squared_distance);
        Eigen::Vector3d pairpos;
        pairpos(0) = goodPts->points[pair_id[0]].x;
        pairpos(1) = goodPts->points[pair_id[0]].y;
        pairpos(2) = goodPts->points[pair_id[0]].z;
        Eigen::Vector3d goodrp = goodPtsPset.find(pairpos)->second; // search in  good points structure and return the position
        pset.row(poorIdx[pp]) = goodrp;
    }
}

void RosaMain::dcrosa() {
    Extra_Del ed_dc;
    Eigen::MatrixXi int_indxs;
    Eigen::MatrixXd newpset, indxs, extract_neighs;
    newpset.resize(pcd_size_, 3);
    Eigen::MatrixXd is_linear;
    is_linear.resize(pcd_size_, 1); 
    is_linear.setConstant(0);

    for (int n=0; n<dcrosa_iter; n++) {
        for (int i=0; i<pcd_size_; i++) {
            if (SSD.neighs[i].size() > 0) {
                // Adjust position according to the mahalanobis neighbors
                int_indxs = Eigen::Map<Eigen::MatrixXi>(SSD.neighs[i].data(), SSD.neighs[i].size(), 1);
                indxs = int_indxs.cast<double>();
                extract_neighs = ed_dc.rows_ext_M(indxs, pset); // Extract the neighbouring ROSA points positions (mahalanobis)
                newpset.row(i) = extract_neighs.colwise().mean(); // Sets each ROSA Point (position) as the mean of the neighbouring points.
            }
            else {
                newpset.row(i) = pset.row(i); // do nothing...
            }
        }

        pset = newpset;

        /* Shrinking */
        pset_cloud->clear();
        pset_cloud->width = pset.rows();
        pset_cloud->height = 1;
        pset_cloud->points.resize(pset_cloud->width * pset_cloud->height);
        
        for (size_t i=0; i<pset_cloud->points.size(); i++) {
            pset_cloud->points[i].x = pset(i,0);
            pset_cloud->points[i].y = pset(i,1);
            pset_cloud->points[i].z = pset(i,2);
        }
        if (pset_cloud->points.empty()) return;

        pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;
        pset_tree.setInputCloud(pset_cloud);

        // Confidence calc
        Eigen::VectorXd conf = Eigen::VectorXd::Zero(pset.rows()); // zero initialized confidence vector
        newpset = pset; 

        double CONFIDENCE_TH = 0.5; // Originally 0.5

        for (int i=0; i<pcd_size_; i++) {
            // std::vector<int> pt_idx(k_KNN);
            // std::vector<float> pt_dists(k_KNN);
            // pset_tree.nearestKSearch(pset_cloud->points[i], k_KNN, pt_idx, pt_dists);

            std::vector<int> pt_idx;
            std::vector<float> pt_dists;
            double pset_rad = 0.005;
            pset_tree.radiusSearch(pset_cloud->points[i], pset_rad, pt_idx, pt_dists);

            Eigen::MatrixXd neighbours(pt_idx.size(), 3);
            for (int j=0; j<pt_idx.size(); j++) {
                neighbours.row(j) = pset.row(pt_idx[j]);
            }

            Eigen::Vector3d local_mean = neighbours.colwise().mean(); // Average of neighbouring ROSA points
            neighbours.rowwise() -= local_mean.transpose(); // Center points around mean

            // The confidence metric is based on the fact that the singular-values represent the variance in the respective 
            // principal direction. If the largest singular value (0) is large relative to the sum, it indicates that
            // the neighbours are highly linear in nature: i.e. skeletonized.
            Eigen::BDCSVD<Eigen::MatrixXd> svd(neighbours, Eigen::ComputeThinU | Eigen::ComputeThinV);
            conf(i) = svd.singularValues()(0) / svd.singularValues().sum();

            if (conf(i) < CONFIDENCE_TH) continue; // If the linearity is not sufficient (curve in skeleton) - dont use the principal axis for linear projection
            
            // Compute linear projection
            // if the neighbouring ROSA points are sufficiently linear, a linear projection of the points onto the principal axis is performed:
            newpset.row(i) = svd.matrixU().col(0).transpose() * (svd.matrixU().col(0) * (pset.row(i) - local_mean.transpose()) ) + local_mean.transpose();
            is_linear(i,0) = 1;
        }
        pset = newpset;
    }

    int new_idx = 0;
    std::unordered_map<int, int> old_to_new_idx;
    Eigen::MatrixXd filtered_pset(static_cast<int>(is_linear.sum()), 3);
    std::vector<std::vector<int>> new_neighs;
    std::vector<std::vector<int>> new_neighs_surf;

    for (int i = 0; i < pcd_size_; ++i) {
        if (is_linear(i, 0) != 1) continue;

        old_to_new_idx[i] = new_idx;
        filtered_pset.row(new_idx) = pset.row(i);

        std::vector<int> filtered_neigh;
        std::vector<int> filtered_neigh_surf;

        for (int old_neighbor : SSD.neighs[i]) {
            auto it = old_to_new_idx.find(old_neighbor);
            if (it != old_to_new_idx.end()) {
                filtered_neigh.push_back(it->second);
            }
        }

        for (int old_neighbor : SSD.neighs_surf[i]) {
            auto it = old_to_new_idx.find(old_neighbor);
            if (it != old_to_new_idx.end()) {
                filtered_neigh_surf.push_back(it->second);
            }
        }

        new_neighs.push_back(filtered_neigh);
        new_neighs_surf.push_back(filtered_neigh_surf);
        new_idx++;
    }
    pset = filtered_pset;
    pcd_size_ = pset.rows();
    SSD.neighs = new_neighs;
    SSD.neighs_surf = new_neighs_surf;
    SSD.pts_matrix = ed_dc.rows_ext_M(is_linear, SSD.pts_matrix);
    SSD.nrs_matrix = ed_dc.rows_ext_M(is_linear, SSD.nrs_matrix);
}

void RosaMain::vertex_sampling() {
    Extra_Del ed_vs;
    int outlier = 5;

    bad_sample = Eigen::MatrixXi::Zero(pcd_size_, 1);
    pcl::PointXYZ pset_pt;
    pset_cloud->clear();
    for (int i=0; i<pcd_size_; i++) {
        // if ((int)SSD.neighs[i].size() <= outlier) {
        //     bad_sample(i,0) = 1;
        // }
        pset_pt.x = pset(i,0);
        pset_pt.y = pset(i,1);
        pset_pt.z = pset(i,2);
        pset_cloud->points.push_back(pset_pt);
    }

    // mindst stores the minimum squared distance from each unassigned point to the nearest assigned skeleton point. 
    Eigen::MatrixXd mindst = Eigen::MatrixXd::Constant(pcd_size_, 1, std::numeric_limits<double>::quiet_NaN()); 
    SSD.corresp = Eigen::MatrixXd::Constant(pcd_size_, 1, -1); // initialized with value -1

    Eigen::MatrixXi int_nidxs;
    Eigen::MatrixXd nIdxs;
    Eigen::MatrixXd extract_corresp;
    pcl::PointXYZ search_point;
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> fps_tree;
    fps_tree.setInputCloud(pset_cloud);
    SSD.skelver.resize(0,3);
    
    sample_radius = leaf_size_ds;
    std::cout << "Vertex Sampling Radius: " << sample_radius << std::endl;

    // Farthest Point Sampling (FPS) / Skeletonization / Vertex selection
    for (int k=0; k<pcd_size_; k++) {
        if (SSD.corresp(k,0) != -1) continue; // skip already assigned points - Will only proceed if gaps larger than search radius in ROSA points (after 1st iter)
        mindst(k,0) = 1e8; // set large to ensure update

        // run while ANY element in corresp is still -1
        while (!((SSD.corresp.array() != -1).all())) {
            int maxIdx = argmax_eigen(mindst); // maxIdx represents the most distant unassigned point

            // If the largest distance value is zero... I.e. all remaining unassinged points are with the radius
            if (!std::isnan(mindst(maxIdx, 0)) && mindst(maxIdx,0) == 0) break;

            // The current search point
            search_point.x = pset(maxIdx,0);
            search_point.y = pset(maxIdx,1);
            search_point.z = pset(maxIdx,2);
            
            // Search for points within the sample_radius of the current search point. 
            // The indices of the nearest points are set in indxs
            indxs.clear();
            radius_squared_distance.clear();
            fps_tree.radiusSearch(search_point, sample_radius, indxs, radius_squared_distance);

            int_nidxs = Eigen::Map<Eigen::MatrixXi>(indxs.data(), indxs.size(), 1); // structures the column vector of the nearest neighbours 
            nIdxs = int_nidxs.cast<double>();
            extract_corresp = ed_vs.rows_ext_M(nIdxs, SSD.corresp); // Extract the section corresp according to the indices of the nearest points

            // If all neighbours wihtin sample_radius already has been assigned (neq to -1) the current point is not needed as vertex
            if ((extract_corresp.array() != -1).all()) {
                mindst(maxIdx,0) = 0;
                continue; // Go to loop start
            }

            // If all neighbours had not been assigned to a corresponding vertex, the current search point is chosen as a new vertex.
            SSD.skelver.conservativeResize(SSD.skelver.rows()+1, SSD.skelver.cols()); // adds one vertex
            SSD.skelver.row(SSD.skelver.rows()-1) = pset.row(maxIdx);

            // for every point withing the sample_radius
            for (int z=0; z<(int)indxs.size(); z++) {

                // if the distance value at this index is unassigned OR if a previous assignment has a larger distance
                // the point is assigned to the new vertex
                // this ensures that every point is assigned to their closest vertex
                if (std::isnan(mindst(indxs[z],0)) || mindst(indxs[z],0) > radius_squared_distance[z]) {
                    mindst(indxs[z],0) = radius_squared_distance[z]; // update minimum distance to closest vertex
                    SSD.corresp(indxs[z], 0) = SSD.skelver.rows() - 1; // Keeps track of which skeleton vertice each point corresponds to (0, 1, 2, 3...)
                }
            }
        }
    }

    int dim = SSD.skelver.rows();
    // std::vector<int> temp_surf(k_KNN);
    std::vector<int> temp_surf;
    std::vector<int> good_neighs;
    SSD.Adj.resize(dim, dim);

    // Create adjacency matrix of the skeleton vertices
    for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
        // if ((int)SSD.neighs[pIdx].size() <= outlier) continue;
        temp_surf.clear();
        good_neighs.clear();
        temp_surf = SSD.neighs_surf[pIdx];
        
        for (int ne=0; ne<(int)temp_surf.size(); ne++) {
            if (bad_sample(temp_surf[ne]) == 0) {
                good_neighs.push_back(temp_surf[ne]);
            }
        }

        for (int nidx=0; nidx<(int)good_neighs.size(); nidx++) {
            SSD.Adj((int)SSD.corresp(pIdx,0), (int)SSD.corresp(good_neighs[nidx],0)) = 1;
            SSD.Adj((int)SSD.corresp(good_neighs[nidx],0), (int)SSD.corresp(pIdx,0)) = 1;
        }
    }
}

void RosaMain::vertex_sampling_kmeans() {
    /* Kmeans clustering for vertex sampling (instead of fps) */
    int n_pts = pset.rows();
    int K = int(norm_scale); // Number of clusters
    Eigen::MatrixXd cluster_centers(K, 3);
    std::vector<int> labels(n_pts, -1);

    std::vector<int> indices(n_pts);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    for (int i=0; i<K; ++i) {
        cluster_centers.row(i) = pset.row(indices[i]);
    }

    int max_iters = 10; 
    for (int iter=0; iter<max_iters; ++iter) {
        bool changed = false;

        for (int i=0; i<n_pts; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            for (int j=0; j<K; ++j) {
                double dist = (pset.row(i) - cluster_centers.row(j)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            if (labels[i] != best_cluster) {
                changed = true;
                labels[i] = best_cluster;
            }
        }

        Eigen::MatrixXd new_centers = Eigen::MatrixXd::Zero(K,3);
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(K);
        for (int i=0; i<n_pts; ++i) {
            new_centers.row(labels[i]) += pset.row(i);
            counts(labels[i]) += 1;
        }

        for (int j=0; j<K; ++j) {
            if (counts[j] > 0) {
                cluster_centers.row(j) = new_centers.row(j) / counts(j);
            }
        }
        if (!changed) break;
    }

    // Assign correspondence of each point
    SSD.skelver = cluster_centers;
    SSD.corresp = Eigen::MatrixXd::Constant(pset.rows(), 1, -1);
    for (int i=0; i<n_pts; ++i) {
        SSD.corresp(i,0) = labels[i];
    }
}

void RosaMain::local_lineextract() {
    /* Edge collapse */
    std::vector<int> ec_neighs;
    Eigen::MatrixXd edge_rows;
    edge_rows.resize(2,3);
    
    while (1) {
        int tricount = 0;
        Eigen::MatrixXi skeds; // Store edges in triangles
        Eigen::MatrixXd skcst; // Store edge distances
        skeds.resize(0,2);
        skcst.resize(0,1);

        for (int i=0; i<SSD.skelver.rows(); i++) {
            ec_neighs.clear();

            for (int col=0; col<SSD.Adj.cols(); col++) {
                if (SSD.Adj(i, col) == 1 && col>i) {
                    ec_neighs.push_back(col);
                }
            }
            for (int j=0; j<(int)ec_neighs.size(); j++) {
                for (int k=j+1; k<(int)ec_neighs.size(); k++) {
                    // If one neighbour of the current vertex is connected to another neighbour of the current vertex
                    // a triangle is detected...
                    if (SSD.Adj(ec_neighs[j], ec_neighs[k]) == 1) {
                        tricount++;

                        // Store the edge between current vertex and the neighbour j.
                        skeds.conservativeResize(skeds.rows()+1, skeds.cols());
                        skeds(skeds.rows()-1, 0) = i;
                        skeds(skeds.rows()-1, 1) = ec_neighs[j];

                        // Store edge length between current vertex and neighbour j.
                        skcst.conservativeResize(skcst.rows()+1, skcst.cols());
                        skcst(skcst.rows()-1, 0) = (SSD.skelver.row(i) - SSD.skelver.row(ec_neighs[j])).norm();

                        // Store the edge between the connected vertice pair 
                        skeds.conservativeResize(skeds.rows()+1, skeds.cols());
                        skeds(skeds.rows()-1, 0) = ec_neighs[j];
                        skeds(skeds.rows()-1, 1) = ec_neighs[k];

                        // Store this edge length
                        skcst.conservativeResize(skcst.rows()+1, skcst.cols());
                        skcst(skcst.rows()-1, 0) = (SSD.skelver.row(ec_neighs[j]) - SSD.skelver.row(ec_neighs[k])).norm();

                        // Store the edge between the current vertex and neighbour k
                        skeds.conservativeResize(skeds.rows()+1, skeds.cols());
                        skeds(skeds.rows()-1, 0) = ec_neighs[k];
                        skeds(skeds.rows()-1, 1) = i;

                        // And the edge length...
                        skcst.conservativeResize(skcst.rows()+1, skcst.cols());
                        skcst(skcst.rows()-1, 0) = (SSD.skelver.row(ec_neighs[k]) - SSD.skelver.row(i)).norm();
                    }
                }
            }
        }

        if (tricount == 0) break;

        Eigen::MatrixXd::Index minRow, minCol;
        skcst.minCoeff(&minRow, &minCol);
        int idx = minRow;
        Eigen::Vector2i edge = skeds.row(idx); // Contains the vertex indices forming the edge of minimum distance...

        edge_rows.row(0) = SSD.skelver.row(edge(0));
        edge_rows.row(1) = SSD.skelver.row(edge(1));
        SSD.skelver.row(edge(0)) = edge_rows.colwise().mean();
        SSD.skelver.row(edge(1)).setConstant(std::numeric_limits<double>::quiet_NaN());

        for(int r=0; r<SSD.corresp.rows(); r++) {
            if (SSD.corresp(r,0) == (double)edge(1)) {
                SSD.corresp(r,0) = (double)edge(0);
            }
        }

        for (int k=0; k<SSD.Adj.rows(); k++) {
            // If any vertex was connected to the redundant vertex in the triangle
            // connect it to the new vertex (mean)
            if (SSD.Adj(edge(1),k) == 1) {
                SSD.Adj(edge(0),k) = 1;
                SSD.Adj(k,edge(0)) = 1;
            }
        }

        SSD.Adj.row(edge(1)) = Eigen::MatrixXi::Zero(1, SSD.Adj.cols());
        SSD.Adj.col(edge(1)) = Eigen::MatrixXi::Zero(SSD.Adj.rows(), 1);
    }
}

void RosaMain::vertex_recenter() {
    Extra_Del ed_rr;
    std::vector<int> idxs;
    std::vector<int> deleted_vertices_idx;
    Eigen::MatrixXi ne_idxs;
    Eigen::MatrixXd ne_idxs_d;
    Eigen::MatrixXi del_idxs;
    Eigen::MatrixXd del_idxs_d;
    Eigen::MatrixXd extract_pts;
    Eigen::MatrixXd extract_nrs;
    Eigen::Vector3d proj_center;
    Eigen::Vector3d eucl_center;
    Eigen::Vector3d fuse_center;

    // Extract points corresponding to each vertex
    // Delete vertices if too few points are assigned that vertex
    for (int i=0; i<SSD.skelver.rows(); i++) {
        idxs.clear();
        for (int j=0; j<SSD.corresp.rows(); j++) {
            if (SSD.corresp(j,0) == (double)i) {
                idxs.push_back(j);
            }
        }

        if (idxs.size() < 5) {
            // Bad vertice - Not enough points "agree" with it
            deleted_vertices_idx.push_back(i);
        }

        else {
            ne_idxs = Eigen::Map<Eigen::MatrixXi>(idxs.data(), idxs.size(), 1);
            ne_idxs_d = ne_idxs.cast<double>();
            extract_pts = ed_rr.rows_ext_M(ne_idxs_d, SSD.pts_matrix);
            extract_nrs = ed_rr.rows_ext_M(ne_idxs_d, SSD.nrs_matrix);
            proj_center = closest_projection_point(extract_pts, extract_nrs);

            if (abs(proj_center(0)) < 1 && abs(proj_center(1)) < 1 && abs(proj_center(2)) < 1) {
                eucl_center = extract_pts.colwise().mean();
                fuse_center = alpha_recenter * proj_center + (1 - alpha_recenter)*eucl_center;
                SSD.skelver(i,0) = fuse_center(0);
                SSD.skelver(i,1) = fuse_center(1);
                SSD.skelver(i,2) = fuse_center(2);
            }
        }
    }

    // Remove invalid vertices
    int del_size = deleted_vertices_idx.size();
    if (del_size > 0) {
        // resize skelver
        del_idxs = Eigen::Map<Eigen::MatrixXi>(deleted_vertices_idx.data(), del_size, 1);
        del_idxs_d = del_idxs.cast<double>();
        SSD.skelver = ed_rr.rows_del_M(del_idxs_d, SSD.skelver); // Delete rows...
    }
}

void RosaMain::restore_scale() {
    Extra_Del ed_rs;
    /* Restore scale and apply trasnform to global frame */

    Eigen::MatrixXd scaled_temp(SSD.skelver.rows(), 3);
    Eigen::RowVector3d centroid3 = centroid.head<3>().transpose();
    int valid_count = 0;

    Eigen::Quaterniond q(transform.transform.rotation.w,
                         transform.transform.rotation.x,
                         transform.transform.rotation.y,
                         transform.transform.rotation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d t(transform.transform.translation.x,
                      transform.transform.translation.y,
                      transform.transform.translation.z);
        
    for (int i=0; i<(int)SSD.skelver.rows(); ++i) {
        Eigen::Vector3d rescl = SSD.skelver.row(i) * norm_scale + centroid3;
        double dist = rescl.norm();
        if (dist >= closest_point) {
            rescl = R * rescl + t;
            scaled_temp.row(valid_count++) = rescl;
        }
        else {
            std::cout << "Remove vertex due to distance constraint..." << std::endl;
        }
    }
    
    // pcl::KdTreeFLANN<pcl::PointXYZ> vscale_tree; 
    // pcl::PointCloud<pcl::PointXYZ>::Ptr vscale_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // for (int i=0; i<(int)scaled_temp.rows(); ++i) {
    //     pcl::PointXYZ pt;
    //     pt.x = scaled_temp(i,0);
    //     pt.y = scaled_temp(i,1);
    //     pt.z = scaled_temp(i,2);
    //     vscale_cloud->points.push_back(pt);
    // }
    
    // vscale_tree.setInputCloud(vscale_cloud);
    // for (int i=0; i<(int)scaled_temp.rows(); ++i) {
    //     std::vector<int> ids;
    //     std::vector<float> dists;
    //     Eigen::MatrixXi ids_temp;
    //     Eigen::MatrixXd ids_d;
    //     vscale_tree.nearestKSearch(vscale_cloud->points[i], k_KNN, ids, dists);
    //     ids_temp = Eigen::Map<Eigen::MatrixXi>(ids.data(), ids.size(), 1);
    //     ids_d = ids_temp.cast<double>();
    //     Eigen::MatrixXd current = ed_rs.rows_ext_M(ids_d, scaled_temp);
    //     auto [dir, lin] = PCA(current);

    // }

    SSD.skelver_scaled = scaled_temp;

    // scaled_temp = scaled_temp.topRows(valid_count);
    // auto [dir, lin] = PCA(scaled_temp);
    // double lin_th = 0.8;

}

void RosaMain::kf_skeleton_incr() {
    for (int i=0; i<SSD.skelver_scaled.rows(); i++) {
        Eigen::Vector3d ver = SSD.skelver_scaled.row(i);
        bool matched = false;

        for (auto &gver : SSD.gskel) {
            // If the vertex is close to a global vertex -- Update using LKF
            if ((gver.position - ver).norm() < kf_dist_th) {
                VertexLKF kf(kf_pn, kf_mn);
                kf.initialize(gver.position, gver.covariance);
                kf.update(ver);

                gver.position = kf.getState();
                gver.covariance = kf.getCovariance();
                gver.observation_count++;

                double trace = gver.covariance.trace();
                if (trace < kf_conf_th) {
                    gver.confidence_check = true;
                }

                matched = true;
                break; // break if close point found...
            }
        }

        // Else introduce new vertex...
        if (!matched) {
            SkeletonVertex new_ver;
            new_ver.position = ver;
            new_ver.covariance = Eigen::Matrix3d::Identity();
            new_ver.observation_count = 1;
            new_ver.confidence_check = false;
            SSD.gskel.push_back(new_ver);
        }
    }

    // If the confidence of a point is not satisfactory within 3 iteration -- Remove it again
    if (kf_cnt == 2) {
        for (int i=0; i<(int)SSD.gskel.size(); ) {
            if (!SSD.gskel[i].confidence_check) {
                SSD.gskel.erase(SSD.gskel.begin() + i); 
                }
            else {
                    i++;
                }
            }

        kf_cnt = 0;
    }
    kf_cnt++;
}

void RosaMain::graph_adj() {
    // Create global skeleton adjacency matrix from points that pass the confidence check
    SSD.gskel_val.clear();
    SSD.ver_cloud->clear();
    int pre_size = SSD.gskel_val.size();

    pcl::PointXYZ pt;
    for (auto &gver : SSD.gskel) {
        if (gver.confidence_check) {
            pt.x = gver.position(0);
            pt.y = gver.position(1);
            pt.z = gver.position(2);
            SSD.ver_cloud->points.push_back(pt);
            SSD.gskel_val.push_back(gver);
        }
    }

    new_vers = SSD.gskel_val.size() - pre_size; // Number of new vertices this iteration (used for mst update)

    if (SSD.ver_cloud->points.empty()) return;

    int dim = SSD.gskel_val.size();
    SSD.gadj = Eigen::MatrixXi::Zero(dim, dim);

    // KNN search for adjacency 
    pcl::KdTreeFLANN<pcl::PointXYZ> adj_tree;
    adj_tree.setInputCloud(SSD.ver_cloud);

    int K = 5;
    double dist_th = 5.0;

    for (int i = 0; i < (int)SSD.ver_cloud->points.size(); i++) {
        std::vector<int> indxs;
        std::vector<float> dists;
        int n_nb = adj_tree.nearestKSearch(SSD.ver_cloud->points[i], K, indxs, dists); // Number of found neighbors
        
        for (int j = 1; j < n_nb; ++j) { // skip SP itself at index 0
            int nb_j = indxs[j];  // neighbor index            
            float dist_SP_to_NB = (SSD.ver_cloud->points[i].getVector3fMap() - SSD.ver_cloud->points[nb_j].getVector3fMap()).norm();  // distance from SP to neighbor

            if (dist_SP_to_NB > dist_th) continue;

            bool sp_is_closest = true;
    
            for (int k = 1; k < n_nb; ++k) {
                if (k == j) continue;
    
                int nb_k = indxs[k];  // another neighbor index
                // Calculate distance between neighbors (NB(j) and NB(k))
                float dist_NB_to_NB = (SSD.ver_cloud->points[nb_j].getVector3fMap() - SSD.ver_cloud->points[nb_k].getVector3fMap()).norm();
                float dist_SP_to_NB2 = (SSD.ver_cloud->points[i].getVector3fMap() - SSD.ver_cloud->points[nb_k].getVector3fMap()).norm();

                // If the distance from SP to NB(j) is greater than the distance between NB(j) and NB(k), 
                // AND the distance from SP to NB(k) is smaller than to NB(j)ver_cloud
                // break -> NB(i) is not a valid nb to SP...
                if (dist_NB_to_NB < dist_SP_to_NB && dist_SP_to_NB > dist_SP_to_NB2) {
                    sp_is_closest = false;
                    break;
                }
            }
    
            // If SP is closer to NB(i) than to any other neighbor, mark them as adjacent
            if (sp_is_closest) {
                SSD.gadj(i, nb_j) = 1;
                SSD.gadj(nb_j, i) = 1;// Mark bidirectional adjacency
            }
        }
    }
}

void RosaMain::global_lineextraction() {
    std::vector<int> rm_idx;
    std::vector<int> ec_neighs;
    int dim = SSD.gadj.rows();

    while (1) {
        int tricount = 0;
        Eigen::MatrixXi skeds; // Store edges in triangles
        Eigen::MatrixXd skcst; // Store edge distances
        skeds.resize(0,2);
        skcst.resize(0,1);

        for (int i=0; i<dim; i++) {
            ec_neighs.clear();

            for (int col=0; col<dim; col++) {
                if (SSD.gadj(i, col) == 1 && col>i) {
                    ec_neighs.push_back(col);
                }
            }
            for (int j=0; j<(int)ec_neighs.size(); j++) {
                for (int k=j+1; k<(int)ec_neighs.size(); k++) {
                    if (SSD.gadj(ec_neighs[j], ec_neighs[k]) == 1) {
                        tricount++;

                        Eigen::Vector3d ver_i = SSD.gskel_val[i].position;
                        Eigen::Vector3d ver_j = SSD.gskel_val[ec_neighs[j]].position;
                        Eigen::Vector3d ver_k = SSD.gskel_val[ec_neighs[k]].position;

                        // Store the edge between current vertex and the neighbour j.
                        skeds.conservativeResize(skeds.rows()+1, skeds.cols());
                        skeds(skeds.rows()-1, 0) = i;
                        skeds(skeds.rows()-1, 1) = ec_neighs[j];

                        // Store edge length between current vertex and neighbour j.
                        skcst.conservativeResize(skcst.rows()+1, skcst.cols());
                        skcst(skcst.rows()-1, 0) = (ver_i - ver_j).norm();

                        // Store the edge between the connected vertice pair 
                        skeds.conservativeResize(skeds.rows()+1, skeds.cols());
                        skeds(skeds.rows()-1, 0) = ec_neighs[j];
                        skeds(skeds.rows()-1, 1) = ec_neighs[k];

                        // Store this edge length
                        skcst.conservativeResize(skcst.rows()+1, skcst.cols());
                        skcst(skcst.rows()-1, 0) = (ver_j - ver_k).norm();

                        // Store the edge between the current vertex and neighbour k
                        skeds.conservativeResize(skeds.rows()+1, skeds.cols());
                        skeds(skeds.rows()-1, 0) = ec_neighs[k];
                        skeds(skeds.rows()-1, 1) = i;

                        // And the edge length...
                        skcst.conservativeResize(skcst.rows()+1, skcst.cols());
                        skcst(skcst.rows()-1, 0) = (ver_k - ver_i).norm();
                    }
                }
            }
        }

        if (tricount == 0) break; // No triangles found...
        Eigen::MatrixXd edge_rows;
        edge_rows.resize(2,3);

        Eigen::MatrixXd::Index minRow, minCol;
        skcst.minCoeff(&minRow, &minCol); // Find the minimum distance edge
        int idx = minRow;
        Eigen::Vector2i edge = skeds.row(idx);
        edge_rows.row(0) = SSD.gskel_val[edge(0)].position;
        edge_rows.row(1) = SSD.gskel_val[edge(1)].position;
        SSD.gskel_val[edge(0)].position = edge_rows.colwise().mean();
        rm_idx.push_back(edge(1));

        // Update adjacency matrix for next iteration...
        for (int k=0; k<SSD.gadj.rows(); k++) {
            if (SSD.gadj(edge(1), k) == 1) {
                SSD.gadj(edge(0), k) = 1;
                SSD.gadj(k, edge(0)) = 1;
            }
        }
        SSD.gadj.row(edge(1)).setZero();
        SSD.gadj.col(edge(1)).setZero();
    }
    
    // remove redundat edges
    std::sort(rm_idx.rbegin(), rm_idx.rend()); // Sort in descending order
    for (int idx : rm_idx) {
        SSD.gskel_val.erase(SSD.gskel_val.begin() + idx);
    }

    // adjust adjacency matrix
    std::vector<int> keep_ids; //indicies to keep
    for (int i=0; i<dim; i++) {
        if (std::find(rm_idx.begin(), rm_idx.end(), i) == rm_idx.end()) {
            keep_ids.push_back(i);
        }
    }
    Eigen::MatrixXi reduced(keep_ids.size(), keep_ids.size());
    for (int i=0; i<(int)keep_ids.size(); i++) {
        for (int j=0; j<(int)keep_ids.size(); j++) {
            reduced(i,j) = SSD.gadj(keep_ids[i], keep_ids[j]);
        }
    }
    SSD.gadj = reduced;

    // Mean recentering...
    for (int i=0; i<(int)SSD.gskel_val.size(); i++) {
        std::vector<int> neighbors;
        for (int j=0; j<SSD.gadj.cols(); j++) {
            if (SSD.gadj(i,j) == 1) {
                neighbors.push_back(j);
            }
        }
        if (!neighbors.empty()) {
            Eigen::Vector3d avg_pos(0,0,0);
            for (int nid : neighbors) {
                avg_pos += SSD.gskel_val[nid].position;
            }
            avg_pos /= neighbors.size();
            SSD.gskel_val[i].position = avg_pos;
        }
    }
}





void RosaMain::mst() {
    int N_ver = SSD.gskel_val.size();
    if (N_ver == 0 || new_vers == 0) return; // No vertices in the skeleton yet or no new vertices to add

    // Step 1: Construct the edge list for the existing skeleton
    // This part takes all the existing edges and stores them with their corresponding distances as weights
    // Only edges that are already connected in the adjacency matrix (SSD.gadj) are considered (prev iteration...)
    std::vector<Edge> mst_edges;
    // for (int i = 0; i < N_ver - new_vers; ++i) {
    int start_idx = N_ver - new_vers;
    for (int i = 0; i < N_ver; ++i) {
    // for (int i=start_idx; i < N_ver; ++i) {
        for (int j = i + 1; j < N_ver; ++j) {
            if (SSD.gadj(i, j) == 1) {
                // Compute the weight of the edge as the Euclidean distance between vertices i and j
                double weight = (SSD.gskel_val[i].position - SSD.gskel_val[j].position).norm();
                // Store the edge and its weight
                mst_edges.push_back({i, j, weight});
            }
        }
    }

    // Step 2: Apply Kruskal’s algorithm to form the MST
    // Sort all edges by their weight
    std::sort(mst_edges.begin(), mst_edges.end());

    // Initialize the Union-Find (Disjoint Set Union) data structure
    UnionFind uf(N_ver); // Initially, each vertex is its own parent, representing a disjoint set

    // Reset the adjacency matrix to zero (no edges initially)
    SSD.gadj.setZero(); 

    // Step 3: Process edges in ascending order of their weight
    // If two vertices belong to different sets (i.e., adding the edge won't form a cycle),
    // then unite them and add the edge to the MST
    for (const auto &edge : mst_edges) {
        // If the vertices u and v are in different sets, unite them and add the edge
        if (uf.unite(edge.u, edge.v)) {
            // Mark this edge as part of the MST in the adjacency matrix (bidirectional connection)
            SSD.gadj(edge.u, edge.v) = 1;
            SSD.gadj(edge.v, edge.u) = 1;
        }
    }
}







void RosaMain::graph_decomp() {
    SSD.joint_ids.clear();
    SSD.end_ids.clear();
    SSD.bad_ids.clear();

    for (int i=0; i<(int)SSD.gadj.rows(); i++) {
        int degree = SSD.gadj.row(i).sum();
        if (degree == 0) {
            // bad vertice (not connected)
            SSD.bad_ids.push_back(i);
        }

        if (degree == 1) {
            SSD.end_ids.push_back(i);
        }

        if (degree > 2) {
            SSD.joint_ids.push_back(i);
        }
    }
}

void RosaMain::vertex_merge() {
    // Merge vertices if... 
        // Two joints are connected
        // Two points are too close

    auto is_joint = [&](int idx) {
        return std::find(SSD.joint_ids.begin(), SSD.joint_ids.end(), idx) != SSD.joint_ids.end();
    };
    
    for (int i=0; i<(int)SSD.gskel_val.size(); ++i) {
        for (int j=i+1; j<(int)SSD.gskel_val.size(); ++j) {
            if (SSD.gadj(i,j) != 1) continue; // Not connected

            bool merge = false;

            // case 1: If two joints are connected
            if (is_joint(i) && is_joint(j)) {
                SSD.joint_ids.erase(std::remove(SSD.joint_ids.begin(), SSD.joint_ids.end(),j), SSD.joint_ids.end());
                merge = true;
            }

            // case 2: If two points are too close
            if (!merge && (SSD.gskel_val[i].position - SSD.gskel_val[j].position).norm() < kf_dist_th) {
                merge = true;
            }
      
            if (merge) {
                int obs_i = SSD.gskel_val[i].observation_count;
                int obs_j = SSD.gskel_val[j].observation_count;
                int total_obs = obs_i + obs_j;
                
                SSD.gskel_val[i].position = (SSD.gskel_val[i].position*obs_i + SSD.gskel_val[j].position*obs_j) / total_obs;
                SSD.gskel_val[i].observation_count = total_obs;

                // Rewire connectios to the merged vertex...
                for (int k=0; k<SSD.gadj.rows(); ++k) {
                    if (SSD.gadj(j,k) == 1) SSD.gadj(i,k) = 1;
                    if (SSD.gadj(k,j) == 1) SSD.gadj(k,i) = 1;
                }

                // Remove vertex j...
                SSD.gskel_val.erase(SSD.gskel_val.begin() + j);
                SSD.gadj.block(j, 0, SSD.gadj.rows()-j-1, SSD.gadj.cols()) = SSD.gadj.block(j+1, 0, SSD.gadj.rows()-j-1, SSD.gadj.cols());
                SSD.gadj.block(0, j, SSD.gadj.rows(), SSD.gadj.cols()-j-1) = SSD.gadj.block(0, j+1, SSD.gadj.rows(), SSD.gadj.cols()-j-1);
                SSD.gadj.conservativeResize(SSD.gadj.rows()-1, SSD.gadj.cols()-1);

                --j; // Stay at same index since matrix is shifted...
            }
        }
    }
}

void RosaMain::branch_extract() {
    float branch_angle_th = 15;
    const double cos_thresh = std::cos(branch_angle_th * M_PI / 180.0);
    std::set<int> visited;
    int branch_id = 0;

    for (int i = 0; i < SSD.gadj.rows(); ++i) {
        if (visited.count(i)) continue;

        for (int j = 0; j < SSD.gadj.cols(); ++j) {
            if (SSD.gadj(i, j) == 1 && !visited.count(j)) {
                // Start a new branch
                std::vector<int> branch = {i};
                visited.insert(i);

                int prev = i;
                int curr = j;
                visited.insert(curr);
                branch.push_back(curr);

                Eigen::Vector3d last_dir = SSD.gskel_val[curr].position - SSD.gskel_val[prev].position;
                last_dir.normalize();

                while (true) {
                    int next = -1;
                    Eigen::Vector3d best_dir;
                    double best_dot = cos_thresh;

                    for (int k = 0; k < SSD.gadj.cols(); ++k) {
                        if (SSD.gadj(curr, k) == 1 && !visited.count(k)) {
                            Eigen::Vector3d dir = SSD.gskel_val[k].position - SSD.gskel_val[curr].position;
                            dir.normalize();
                            double dot = last_dir.dot(dir);

                            if (dot > best_dot) {
                                next = k;
                                best_dot = dot;
                                best_dir = dir;
                            }
                        }
                    }

                    if (next == -1) break;

                    prev = curr;
                    curr = next;
                    last_dir = best_dir;
                    visited.insert(curr);
                    branch.push_back(curr);
                }

                if (branch.size() > 1) {
                    SSD.branches[branch_id++] = branch;
                }
            }
        }
    }
}

void RosaMain::prune_branches() {
    int min_branch_size = 2;
    int N_ver = SSD.gadj.rows();
    std::vector<int> visited(N_ver, false);
    
    for (int i=0; i<N_ver; ++i) {
        int degree = SSD.gadj.row(i).sum();
        if (degree == 0) continue; // No neighbors
        
        for (int j=0; j<N_ver; ++j) {
            if (SSD.gadj(i,j) == 1 && !visited[j]) {
                std::vector<int> branch = dfs_branch_collect(j,i);

                for (int b : branch) visited[b] = true;

                // Remove connections for small branches...
                if ((int)branch.size() < min_branch_size) {
                    for (int b : branch) {
                        for (int k=0; k<N_ver; ++k) {
                            SSD.gadj(b,k) = 0;
                            SSD.gadj(k,b) = 0;
                        }
                    }
                }
            }
        }
    }
}

void RosaMain::update_skeleton() {
    // Add vertices to final cloud...
    SSD.global_skeleton->clear();
    pcl::PointXYZ pt;
    for (int i=0; i<(int)SSD.gskel_val.size(); i++) {
        if (SSD.gadj.row(i).sum() > 0) {
            pt.x = SSD.gskel_val[i].position[0];
            pt.y = SSD.gskel_val[i].position[1];
            pt.z = SSD.gskel_val[i].position[2];
            SSD.global_skeleton->points.push_back(pt);
        }
    }
}



/* Helper Functions */

void RosaMain::rosa_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals) {
    Eigen::Matrix3d M;
    Eigen::Vector3d normal_v;
    pset.resize(pcd_size_, 3);
    vset.resize(pcd_size_, 3);
    vvar.resize(pcd_size_, 1);
    for (int i=0; i<pcd_size_; i++) {
        pset(i,0) = cloud->points[i].x;
        pset(i,1) = cloud->points[i].y;
        pset(i,2) = cloud->points[i].z;
        normal_v(0) = normals->points[i].normal_x;
        normal_v(1) = normals->points[i].normal_y;
        normal_v(2) = normals->points[i].normal_z;
        M = create_orthonormal_frame(normal_v); 
        vset.row(i) = M.row(1); // Extracts a vector orthogonal to normal_v... i.e. a vector that lies in the tangent plane of the structure-surface.
    }
}

Eigen::Matrix3d RosaMain::create_orthonormal_frame(Eigen::Vector3d &v) {

     /* random process for generating orthonormal basis */
     v = v/v.norm();
     double TH_ZERO = 1e-10;
    //  srand((unsigned)time(NULL));

     Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
     M(0,0) = v(0); 
     M(0,1) = v(1); 
     M(0,2) = v(2);
     Eigen::Vector3d new_vec, temp_vec;
 
     // The outer for loops finds an orthonormal basis
     for (int i=1; i<3; ++i) {
       new_vec.setRandom();
       new_vec = new_vec/new_vec.norm();
 
       while (abs(1.0 - v.dot(new_vec)) < TH_ZERO) {
         // Run until vector (not too parallel) is found... Avoid colinear vectors
         new_vec.setRandom();
         new_vec = new_vec / new_vec.norm();
       }

       // Gramm-Schmidt process to find orthogonal vectors
       for (int j=0; j<i; ++j) {
         temp_vec = (new_vec - new_vec.dot(M.row(j)) * (M.row(j).transpose()));
         new_vec = temp_vec/temp_vec.norm();
       }
 
       M(i,0) = new_vec(0);
       M(i,1) = new_vec(1);
       M(i,2) = new_vec(2);
     }
 
     return M;
}

Eigen::MatrixXd RosaMain::compute_active_samples(int &idx, Eigen::Vector3d &p_cut, Eigen::Vector3d &v_cut) {
    // Extracts an index-vector masked with the indices on the plane slice
    Eigen::MatrixXd out_indxs(pcd_size_, 1);
    int out_size = 0;
    std::vector<int> isoncut(pcd_size_, 0); // On cut mask

    std::vector<double> p(3); // Current point
    p[0] = p_cut(0);
    p[1] = p_cut(1);
    p[2] = p_cut(2);
    std::vector<double> n(3); // Corresponding plane normal vector
    n[0] = v_cut(0);
    n[1] = v_cut(1);
    n[2] = v_cut(2);

    std::vector<double> Pi(3); // Point to check if isoncut
    for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
        Pi[0] = SSD.pts_->points[pIdx].x;
        Pi[1] = SSD.pts_->points[pIdx].y;
        Pi[2] = SSD.pts_->points[pIdx].z;

        // Determine if the current point is included in the plane slice. That is within delta distance from the plane...
        // Distance is calculated as d = n*(p - P)
        // using the plane equation: https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx
        if (fabs(n[0]*(p[0]-Pi[0]) + n[1]*(p[1]-Pi[1]) + n[2]*(p[2]-Pi[2])) < delta) {
            isoncut[pIdx] = 1;
        }
    }

    // Flood-fill algorithm to ensure that the other regions of plane intersection is not included...
    std::vector<int> queue;
    queue.reserve(pcd_size_); // Allocate memory
    queue.emplace_back(idx); // Insert the seed-point for region growing

    int curr;
    while (!queue.empty()) {
        curr = queue.back();
        queue.pop_back();
        isoncut[curr] = 2;
        out_indxs(out_size++, 0) = curr; //Add to final output... 

        // For the current point iterate through its maha neighs...
        for (size_t i = 0; i < SSD.neighs[curr].size(); ++i) {
            // If a maha nb is on-cut...
            if (isoncut[SSD.neighs[curr][i]] == 1) {
                isoncut[SSD.neighs[curr][i]] = 3; // Mark as part of the region
                queue.emplace_back(SSD.neighs[curr][i]); // Set next search point
            }
        }
    }
    out_indxs.conservativeResize(out_size, 1); // Reduces the size down to an array of indices corresponding to the active samples
    return out_indxs;
}

Eigen::Vector3d RosaMain::compute_symmetrynormal(Eigen::MatrixXd& local_normals) {
    // This function determines the vector that minimizes the variance of the angle between local normals and the vector.
    // This can be interpreted as the "direction" of the skeleton inside the structure...
    // The symmetry normal will be the normal vector of the best fit plane of points corresponding to the local_normals

    Eigen::Matrix3d M; Eigen::Vector3d vec;
    int size = local_normals.rows();
    double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;

    // Variances: Computing the mean squared value and substracting the mean squared value -> Variance = E[X²] - E[X]²
    Vxx = local_normals.col(0).cwiseAbs2().sum() / size - pow(local_normals.col(0).sum(), 2) / pow(size, 2);
    Vyy = local_normals.col(1).cwiseAbs2().sum() / size - pow(local_normals.col(1).sum(), 2) / pow(size, 2);
    Vzz = local_normals.col(2).cwiseAbs2().sum() / size - pow(local_normals.col(2).sum(), 2) / pow(size, 2);

    // Covariances: Computing the mean of the product of 2 components and subtracting the product of the means of each components -> Covariance = E[XY] - E[X]E[Y]
    Vxy = 2*(local_normals.col(0).cwiseProduct(local_normals.col(1))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(1).sum()/pow(size, 2);
    Vyx = Vxy;
    Vxz = 2*(local_normals.col(0).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(2).sum()/pow(size, 2);
    Vzx = Vxz;
    Vyz = 2*(local_normals.col(1).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(1).sum()*local_normals.col(2).sum()/pow(size, 2);
    Vzy = Vyz;
    M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

    // Perform singular-value-decomposition on the Covariance matrix M = U(Sigma)V^T
    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();
    // The last column of the matrix U corresponds to the smallest singular value (in Sigma)
    // This in turn represents the direction of smallest variance
    // I.e. for the plance slice -> plane normal. 
    vec = U.col(M.cols()-1);
    return vec;
}

double RosaMain::symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals) {
    // Computes the variance of the local normal vectors projected onto a symmetric normal vector
    Eigen::VectorXd alpha;
    int num = local_normals.rows();

    // calculate the projection of each local normal on the symmetry normal... 
    alpha = local_normals * symm_nor; // Inner product between the symm_nor and each row (normal) in local_normals
    
    // Calculate sample variance of the projections
    double var;
    var = alpha.squaredNorm() / num - pow(alpha.mean(), 2); // sum(alphas)/N - mean(alpha)²
    if (num > 1) {
        var /= (num - 1.0); // *1/(N-1)
    }
    return var;
}

Eigen::Vector3d RosaMain::symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w) {
    // V: vset_ex = symmetry normals computed for the neighbours of a point
    // w: vvar_ex = reciprocal variances (fourth power) of local normal projections on symmetry normal

    Eigen::Matrix3d M; 
    Eigen::Vector3d vec;
    double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;

    // Variances: Summation of the elemet wise product (inner product) between variance and the squared abs value of the
    // sum(w(i)*V(i)²) --- Where V is either x,y, or z component of symmetry normal vector
    Vxx = (w.cwiseProduct(V.col(0).cwiseAbs2())).sum();
    Vyy = (w.cwiseProduct(V.col(1).cwiseAbs2())).sum();
    Vzz = (w.cwiseProduct(V.col(2).cwiseAbs2())).sum();

    // Covariances: Similarly
    // sum(w(i)*Vx(i)*Vy(i)) etc..
    Vxy = (w.cwiseProduct(V.col(0)).cwiseProduct(V.col(1))).sum();
    Vyx = Vxy;
    Vxz = (w.cwiseProduct(V.col(0)).cwiseProduct(V.col(2))).sum();
    Vzx = Vxz;
    Vyz = (w.cwiseProduct(V.col(1)).cwiseProduct(V.col(2))).sum();
    Vzy = Vyz;
    M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

    // The variances are reciprocal fourth order meaning large variances contribute with smaller values in the summation...
    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();

    // The vector corresponding to the largest singular value (first column of U)
    // It represents the the vector of smallest variance amongst the symmetry normals in the neighbourhood of the current point.
    vec = U.col(0);

    return vec;
}

Eigen::Vector3d RosaMain::closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V) {
    // Takes points (P) and corresponding surface normal vectors (V)
    // Each P and corresponding V defines an implicit plane equations: Vi(X-Pi)=0 representing all points on the plane passing through Pi with normal Vi
    // Goal is to find a single point X* that is as close as possible to all these planes (in a least squares sense)
    Eigen::Vector3d vec;
    Eigen::VectorXd Lix2, Liy2, Liz2;

    // Squared components of V
    Lix2 = V.col(0).cwiseAbs2();
    Liy2 = V.col(1).cwiseAbs2();
    Liz2 = V.col(2).cwiseAbs2();

    // Formulate the linear system MX = B
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    Eigen::Vector3d B = Eigen::Vector3d::Zero();

    M(0,0) = (Liy2+Liz2).sum(); // sum(Viy²+Viz²)
    M(0,1) = -(V.col(0).cwiseProduct(V.col(1))).sum(); // -sum(Vix*Viy)
    M(0,2) = -(V.col(0).cwiseProduct(V.col(2))).sum(); // -sum(Vix*Viz)

    M(1,0) = -(V.col(1).cwiseProduct(V.col(0))).sum(); // -sum(Viy*Vix)
    M(1,1) = (Lix2 + Liz2).sum(); // sum(Vix²+Viz²)
    M(1,2) = -(V.col(1).cwiseProduct(V.col(2))).sum(); // -sum(Viy*Viz)

    M(2,0) = -(V.col(2).cwiseProduct(V.col(0))).sum(); // -sum(Viz*Vix)
    M(2,1) = -(V.col(2).cwiseProduct(V.col(1))).sum(); // -sum(Viz*Viy)
    M(2,2) = (Lix2 + Liy2).sum(); // sum(Vix²+Viy²)

    // sum( Pix(Viy²+Viz²) - PiyVixViy - PizVixViz )
    B(0) = (P.col(0).cwiseProduct(Liy2 + Liz2)).sum() - (V.col(0).cwiseProduct(V.col(1)).cwiseProduct(P.col(1))).sum() - (V.col(0).cwiseProduct(V.col(2)).cwiseProduct(P.col(2))).sum();
    // sum( Piy(Vix²+Viz²) - PixVixViy - PizViyViz )
    B(1) = (P.col(1).cwiseProduct(Lix2 + Liz2)).sum() - (V.col(1).cwiseProduct(V.col(0)).cwiseProduct(P.col(0))).sum() - (V.col(1).cwiseProduct(V.col(2)).cwiseProduct(P.col(2))).sum();
    // sum( Piz(Vix²+Viy²) - PixVixViz - PiyViyViz )
    B(2) = (P.col(2).cwiseProduct(Lix2 + Liy2)).sum() - (V.col(2).cwiseProduct(V.col(0)).cwiseProduct(P.col(0))).sum() - (V.col(2).cwiseProduct(V.col(1)).cwiseProduct(P.col(1))).sum();

    if (std::abs(M.determinant()) < 1e-3) {
        vec << 1e8, 1e8, 1e8;
    }
    else {
        // Solving a least squares minimization problem to find the best fit projection point
        // X = M^(-1) * B
        vec = M.inverse()*B;
    }

    // Use LDLT or LLT for symmetric positive semi-definite systems
    // Eigen::LLT<Eigen::Matrix3d> solver(M);
    // if (solver.info() != Eigen::Success) {
    //     X << 1e8, 1e8, 1e8; // Return dummy if M is not positive definite
    // } else {
    //     X = solver.solve(B);
    // }

    return vec;
}

int RosaMain::argmax_eigen(Eigen::MatrixXd &x) {
    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow,&maxCol);
    int idx = maxRow;
    return idx;
}

void RosaMain::extract_seg_dfs(int current, int parent, std::vector<int> &visited, std::vector<int> &seg) {
    visited[current] = true; // Mark the current as visited
    seg.push_back(current); // append the current to the segment

    for (int k=0; k<SSD.gadj.cols(); ++k) {
        // if current is connected, not the parent, and not visited...
        if (SSD.gadj(current, k) == 1 && k != parent && !visited[k]) {
            extract_seg_dfs(k, current, visited, seg); // Recursion...
        }
    }
}

std::vector<int> RosaMain::dfs_branch_collect(int start, int parent) {
    std::vector<int> stack = {start};
    std::vector<int> branch_nodes;
    std::unordered_set<int> visited_local;

    while (!stack.empty()) {
        int curr = stack.back();
        stack.pop_back();

        if (visited_local.count(curr)) continue; // If current is already in visited_local
        visited_local.insert(curr);
        branch_nodes.push_back(curr);

        for (int i = 0; i < SSD.gadj.rows(); ++i) {
            if (SSD.gadj(curr, i) == 1 && i != parent && !visited_local.count(i)) {
                stack.push_back(i);
            }
        }
    }

    return branch_nodes;
}

std::pair<Eigen::Vector3d, double> RosaMain::PCA(Eigen::MatrixXd& A)
  {
    Eigen::Vector3d vec, vals;
    double linearity;
    Eigen::Vector3d centroid = A.colwise().mean();
    Eigen::Matrix3d cov = (A.rowwise() - centroid.transpose()).transpose() * (A.rowwise() - centroid.transpose()) / double(A.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    vec = eig.eigenvectors().col(2).normalized();
    vals = eig.eigenvalues();
    linearity = vals(2) / vals.sum();

    return {vec, linearity}; // Return the principal direction and the linearity measure...
  }


pcl::PointCloud<pcl::PointXYZ>::Ptr RosaMain::scale_transform_debugger(Eigen::MatrixXd &points) {
    /* For restoring matrix structure to pointcloud */
    pcl::PointCloud<pcl::PointXYZ>::Ptr scaled_(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::RowVector3d centroid3 = centroid.head<3>().transpose();
    Eigen::Quaterniond q(transform.transform.rotation.w,
                         transform.transform.rotation.x,
                         transform.transform.rotation.y,
                         transform.transform.rotation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d t(transform.transform.translation.x,
                      transform.transform.translation.y,
                      transform.transform.translation.z);

    for (int i=0; i<(int)points.rows(); ++i) {
        Eigen::Vector3d pt_local = points.row(i) * norm_scale + centroid3;
        Eigen::Vector3d tfpt = R * pt_local + t;
        pcl::PointXYZ pt;
        pt.x = tfpt(0);
        pt.y = tfpt(1);
        pt.z = tfpt(2);
        scaled_->points.push_back(pt);
    }
    return scaled_;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr RosaMain::scale_transform_debugger(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    /* For restoring point cloud */
    pcl::PointCloud<pcl::PointXYZ>::Ptr scaled_(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::RowVector3d centroid3 = centroid.head<3>().transpose();
    Eigen::Quaterniond q(transform.transform.rotation.w,
                         transform.transform.rotation.x,
                         transform.transform.rotation.y,
                         transform.transform.rotation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d t(transform.transform.translation.x,
                      transform.transform.translation.y,
                      transform.transform.translation.z);

    for (const auto pt : cloud->points) {
        Eigen::Vector3d ptt(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_local = ptt.transpose() * norm_scale + centroid3;
        Eigen::Vector3d tfpt = R * pt_local + t;
        pcl::PointXYZ pt_tf;
        pt_tf.x = tfpt(0);
        pt_tf.y = tfpt(1);
        pt_tf.z = tfpt(2);
        scaled_->points.push_back(pt_tf);
    }
    return scaled_;
}

Eigen::MatrixXd RosaMain::scale_transform_debugger_matmat(Eigen::MatrixXd &points) {
    /* For restoring matrix structure to matrix */
    Eigen::MatrixXd scaled_;
    scaled_.resize(points.rows(), points.cols());

    Eigen::RowVector3d centroid3 = centroid.head<3>().transpose();
    Eigen::Quaterniond q(transform.transform.rotation.w,
                         transform.transform.rotation.x,
                         transform.transform.rotation.y,
                         transform.transform.rotation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d t(transform.transform.translation.x,
                      transform.transform.translation.y,
                      transform.transform.translation.z);

    for (int i=0; i<(int)points.rows(); ++i) {
        Eigen::Vector3d pt_local = points.row(i) * norm_scale + centroid3;
        Eigen::Vector3d tfpt = R * pt_local + t;
        scaled_.row(i) = tfpt;
    }
    return scaled_;
}
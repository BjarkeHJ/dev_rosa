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

    /* Initialize data structures */
    SSD.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.normals_.reset(new pcl::PointCloud<pcl::Normal>);
    SSD.cloud_w_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
    pset_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SSD.rosa_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);

    temp_ver_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    debug_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    debug_cloud_2.reset(new pcl::PointCloud<pcl::PointXYZ>);

    th_mah = 0.1 * radius_neigh;
}

void RosaMain::main() {
    auto start = std::chrono::high_resolution_clock::now();
    distance_filter();
    pcd_size_ = SSD.pts_->points.size();
    if (!pcd_size_) return; // No points within range...

    normalize();
    if (pcd_size_ < nMin) {
        std::cout << "Point Cloud size below nMin... Skipping..." << std::endl; 
        return; // Too few points to reliably compute ROSA pts
    }

    debug_cloud = SSD.pts_;
    mahanalobis_mat(radius_neigh);
    drosa();
    dcrosa();
    vertex_sampling();
    vertex_recenter();
    restore_scale();

    // lineextract();
    // recenter();

    debug_cloud_2 = SSD.rosa_pts;
    // debug_cloud_2 = temp_ver_cloud; // Vertices after FPS

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}

void RosaMain::distance_filter() {
    /* Distance Filtering */
    ptf.setInputCloud(SSD.pts_);
    ptf.setFilterFieldName("x");
    ptf.setFilterLimits(-pts_dist_lim, pts_dist_lim);
    ptf.filter(*SSD.pts_);
    ptf.setFilterFieldName("y");
    ptf.setFilterLimits(-pts_dist_lim, pts_dist_lim);
    ptf.filter(*SSD.pts_);
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
    if (SSD.pts_->points.empty()) {
        std::cout << "Filtered out all points - Cloud empty" << std::endl;
        return;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    SSD.normals_->clear();
    ne.setInputCloud(SSD.pts_);
    ne.setSearchMethod(tree);
    ne.setKSearch(ne_KNN);
    ne.compute(*SSD.normals_); 

    pcl::concatenateFields(*SSD.pts_, *SSD.normals_, *SSD.cloud_w_normals);

    /* Dynamic Voxel Grid Downsampling */
    leaf_size_ds = 0.01;
    while (pcd_size_ > nMax) {
        vgf.setInputCloud(SSD.cloud_w_normals);
        vgf.setLeafSize(leaf_size_ds, leaf_size_ds, leaf_size_ds);
        vgf.filter(*SSD.cloud_w_normals);
        pcd_size_ = SSD.cloud_w_normals->points.size();
        if (pcd_size_ <= nMax) break;
        leaf_size_ds += 0.001;
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
        // Friday 28/3: changed sign as it was missed prev
        nrm.normal_x = -SSD.cloud_w_normals->points[i].normal_x;
        nrm.normal_y = -SSD.cloud_w_normals->points[i].normal_y;
        nrm.normal_z = -SSD.cloud_w_normals->points[i].normal_z;
        SSD.pts_->points.push_back(pt);
        SSD.normals_->points.push_back(nrm);
        SSD.pts_matrix(i,0) = pt.x;
        SSD.pts_matrix(i,1) = pt.y;
        SSD.pts_matrix(i,2) = pt.z;
        SSD.nrs_matrix(i,0) = nrm.normal_x;
        SSD.nrs_matrix(i,1) = nrm.normal_y;
        SSD.nrs_matrix(i,2) = nrm.normal_z;
    }

    th_mah = 0.1*radius_neigh; // Threshold for similarity neighbour extraction
    delta = leaf_size_ds; // Plane slice thickness kept equal to the voxel leaf size
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

void RosaMain::drosa() {
    Extra_Del ed_;
    rosa_initialize(SSD.pts_, SSD.normals_);

    std::vector<std::vector<int>>().swap(SSD.neighs_surf);
    std::vector<int> temp_surf(k_KNN);
    std::vector<float> nn_squared_distance(k_KNN);
    pcl::PointXYZ search_pt_surf;

    // K Nearest neighbours search...
    surf_kdtree.setInputCloud(SSD.pts_);
    for (int i=0; i<pcd_size_; i++) {
        std::vector<int>().swap(temp_surf);
        std::vector<float>().swap(nn_squared_distance);
        search_pt_surf = SSD.pts_->points[i];
        surf_kdtree.nearestKSearch(search_pt_surf, k_KNN, temp_surf, nn_squared_distance);
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
        var_v_p = vset.row(pIdx);
        indxs_p = compute_active_samples(pIdx, var_p_p, var_v_p); // Extract active samples

        /* Update neighbours to be from the same plane slice */
        std::vector<int> temp_neigh;
        for (int p=0; p<(int)indxs_p.rows(); p++) {
            temp_neigh.push_back(indxs_p(p,0));
        }

        SSD.neighs_new.push_back(temp_neigh);

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
            poorIdx.push_back(pIdx);
        }
    }

    /* Reposition poor point to the nearest good point */
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

    for (int n=0; n<dcrosa_iter; n++) {
        for (int i=0; i<pcd_size_; i++) {
            if (SSD.neighs[i].size() > 0) {
                int_indxs = Eigen::Map<Eigen::MatrixXi>(SSD.neighs[i].data(), SSD.neighs[i].size(), 1);
                indxs = int_indxs.cast<double>();
                extract_neighs = ed_dc.rows_ext_M(indxs, pset); // Extract the neighbouring ROSA points positions
                newpset.row(i) = extract_neighs.colwise().mean(); // Sets each ROSA Point (position) as the mean of the neighbouring points.
            }
            else {
                newpset.row(i) = pset.row(i); // do nothing...
            }
        }

        pset = newpset;

        /* Shrinking */
        // pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pset_cloud->clear();
        pset_cloud->width = pset.rows();
        pset_cloud->height = 1;
        pset_cloud->points.resize(pset_cloud->width * pset_cloud->height);
        
        for (size_t i=0; i<pset_cloud->points.size(); i++) {
            pset_cloud->points[i].x = pset(i,0);
            pset_cloud->points[i].y = pset(i,1);
            pset_cloud->points[i].z = pset(i,2);
        }

        pset_tree.setInputCloud(pset_cloud);
        
        // Confidence calc
        Eigen::VectorXd conf = Eigen::VectorXd::Zero(pset.rows()); // zero initialized confidence vector
        newpset = pset; 
        double CONFIDENCE_TH = 0.5; // Originally 0.5

        for (int i=0; i<pcd_size_; i++) {
            std::vector<int> pt_idx(k_KNN);
            std::vector<float> pt_dists(k_KNN);
            pset_tree.nearestKSearch(pset_cloud->points[i], k_KNN, pt_idx, pt_dists);

            Eigen::MatrixXd neighbours(k_KNN, 3);
            for (int j=0; j<k_KNN; j++) {
                neighbours.row(j) = pset.row(pt_idx[j]);
            }

            Eigen::Vector3d local_mean = neighbours.colwise().mean(); // Average of neighbouring ROSA points
            neighbours.rowwise() -= local_mean.transpose(); // Center points around mean

            // The confidence metric is based on the fact that the singular-values represent the variance in the respective 
            // principal direction. If the largest singular value (0) is large relative to the sum, it indicates that
            // the neighbours are highly linear in nature: i.e. skeletonized.
            Eigen::BDCSVD<Eigen::MatrixXd> svd(neighbours, Eigen::ComputeThinU | Eigen::ComputeThinV);
            conf(i) = svd.singularValues()(0) / svd.singularValues().sum();
            if (conf(i) < CONFIDENCE_TH) continue; // Should the sign here not be ">"????
            
            // Compute linear projection
            // if the neighbouring ROSA points are not linear enough, a linear projection is performed:
            // The direction with least variance (dominant singular vector) is the one with the largest singular value (0)
            // The points are then projected onto the dominant direction in the neighbourhood
            newpset.row(i) = svd.matrixU().col(0).transpose() * (svd.matrixU().col(0) * (pset.row(i) - local_mean.transpose()) ) + local_mean.transpose();
        }
        pset = newpset;
    }
}

void RosaMain::vertex_sampling() {
    Extra_Del ed_le;

    int outlier = 2;
    Eigen::MatrixXi bad_sample = Eigen::MatrixXi::Zero(pcd_size_, 1);
    
    pcl::PointXYZ pset_pt;
    pset_cloud->clear();
    for (int i=0; i<pcd_size_; i++) {
        if ((int)SSD.neighs[i].size() <= outlier) {
            bad_sample(i,0) = 1;
        }

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
    fps_tree.setInputCloud(pset_cloud);
    SSD.skelver.resize(0,3);

    // Farthest Point Sampling (FPS) / Skeletonization / Vertex selection
    for (int k=0; k<pcd_size_; k++) {
        if (SSD.corresp(k,0) != -1) continue; // skip already assigned points - Will only proceed if gaps larger than search radius in ROSA points (after 1st iter)
        mindst(k,0) = 1e8; // set large to ensure update

        // run while ANY element in corresp is still -1
        while (!((SSD.corresp.array() != -1).all())) {
            int maxIdx = argmax_eigen(mindst); // maxIdx represents the most distant unassigned point

            if (mindst(maxIdx,0) == 0) break; // If the largest distance value is zero... I.e. all remaining unassinged points are with the radius
            if (!std::isnan(mindst(maxIdx, 0)) && mindst(maxIdx,0) == 0) break;

            // The current search point
            search_point.x = pset(maxIdx,0);
            search_point.y = pset(maxIdx,1);
            search_point.z = pset(maxIdx,2);
            
            // Search for points within the sample_radius of the current search point. 
            // The indices of the nearest points are set in indxs
            std::vector<int>().swap(indxs);
            std::vector<float>().swap(radius_squared_distance);
            fps_tree.radiusSearch(search_point, sample_radius, indxs, radius_squared_distance);

            int_nidxs = Eigen::Map<Eigen::MatrixXi>(indxs.data(), indxs.size(), 1); // structures the column vector of the nearest neighbours 
            nIdxs = int_nidxs.cast<double>();
            extract_corresp = ed_le.rows_ext_M(nIdxs, SSD.corresp); // Extract the section corresp according to the indices of the nearest points

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
    Eigen::MatrixXd temp_skelver;

    // Extract points corresponding to each vertex
    // Delete vertices if too few points are assigned that vertex
    for (int i=0; i<SSD.skelver.rows(); i++) {
        idxs.clear();
        for (int j=0; j<SSD.corresp.rows(); j++) {
            if (SSD.corresp(j,0) == (double)i) {
                idxs.push_back(j);
            }
        }
        if (idxs.size() < 3) {
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
    if (!deleted_vertices_idx.empty()) {
        int del_size = deleted_vertices_idx.size();
        del_idxs = Eigen::Map<Eigen::MatrixXi>(deleted_vertices_idx.data(), del_size, 1);
        del_idxs_d = del_idxs.cast<double>();
        temp_skelver = ed_rr.rows_del_M(del_idxs_d, SSD.skelver); // Delete rows...
        SSD.skelver = temp_skelver;
        SSD.vertices = SSD.skelver;
    }
    else {
        SSD.vertices = SSD.skelver;
    }
}

// Refine local points here... 
// Do lineextract, recenter etc in global refinement

// Maybe lineextaction should be performed on the entire incremented skeleton??
void RosaMain::lineextract() {
    Extra_Del ed_le;
    int outlier = 2;

    pset_cloud->clear();
    pcl::PointXYZ pset_pt;

    Eigen::MatrixXi bad_sample = Eigen::MatrixXi::Zero(pcd_size_, 1); // Bad samples zero initialized
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    pcl::PointXYZ search_point;
    Eigen::MatrixXi int_nidxs;
    Eigen::MatrixXd nIdxs, extract_corresp;
    
    for (int i=0; i<pcd_size_; i++) {
        if ((int)SSD.neighs[i].size() <= outlier) {
            // if the point has less than or equal to 2 neighbours it is classified as a bad_sample
            bad_sample(i,0) = 1;
        }
    }

    for (int j=0; j<pset.rows(); j++) {
        pset_pt.x = pset(j,0);
        pset_pt.y = pset(j,1);
        pset_pt.z = pset(j,2);
        pset_cloud->points.push_back(pset_pt);
    }

    if (pset_cloud->empty()) return;

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(pset_cloud);
    SSD.skelver.resize(0,3);

    // mindst stores the minimum squared distance from each unassigned point to the nearest assigned skeleton point. 
    Eigen::MatrixXd mindst = Eigen::MatrixXd::Constant(pcd_size_, 1, std::numeric_limits<double>::quiet_NaN()); 
    SSD.corresp = Eigen::MatrixXd::Constant(pcd_size_, 1, -1); // initialized with value -1

    // Farthest Point Sampling (FPS) / Skeletonization
    for (int k=0; k<pcd_size_; k++) {
        if (SSD.corresp(k,0) != -1) continue; // skip already assigned points - Will only proceed if gaps larger than search radius in ROSA points (after 1st iter)
        mindst(k,0) = 1e8; // set large to ensure update

        // run while ANY element in corresp is still -1
        while (!((SSD.corresp.array() != -1).all())) {
            int maxIdx = argmax_eigen(mindst); // maxIdx represents the most distant unassigned point

            if (mindst(maxIdx,0) == 0) break; // If the largest distance value is zero... I.e. all remaining unassinged points are with the radius
            if (!std::isnan(mindst(maxIdx, 0)) && mindst(maxIdx,0) == 0) break;

            // The current search point
            search_point.x = pset(maxIdx,0);
            search_point.y = pset(maxIdx,1);
            search_point.z = pset(maxIdx,2);
            
            // Search for points within the sample_radius of the current search point. 
            // The indices of the nearest points are set in indxs
            std::vector<int>().swap(indxs);
            std::vector<float>().swap(radius_squared_distance);
            tree.radiusSearch(search_point, sample_radius, indxs, radius_squared_distance);

            int_nidxs = Eigen::Map<Eigen::MatrixXi>(indxs.data(), indxs.size(), 1); // structures the column vector of the nearest neighbours 
            nIdxs = int_nidxs.cast<double>();
            extract_corresp = ed_le.rows_ext_M(nIdxs, SSD.corresp); // Extract the section corresp according to the indices of the nearest points

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

    temp_ver_cloud->clear();
    for (int i=0; i<(int)SSD.skelver.rows(); i++) {
        pcl::PointXYZ ptt;
        ptt.x = SSD.skelver(i,0) * norm_scale + centroid(0);
        ptt.y = SSD.skelver(i,1) * norm_scale + centroid(1);
        ptt.z = SSD.skelver(i,2) * norm_scale + centroid(2);
        temp_ver_cloud->points.push_back(ptt);
    }


    int dim = SSD.skelver.rows(); // number of vertices
    Eigen::MatrixXi Adj;
    Adj = Eigen::MatrixXi::Zero(dim, dim);
    std::vector<int> temp_surf(k_KNN);
    std::vector<int> good_neighs;

    for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
        temp_surf.clear();
        good_neighs.clear();
        temp_surf = SSD.neighs_surf[pIdx];

        for (int ne=0; ne<(int)temp_surf.size(); ne++) {
            if (bad_sample(temp_surf[ne], 0) == 0) {
                good_neighs.push_back(temp_surf[ne]);
            }
        }

        if (SSD.corresp(pIdx,0) == -1) continue; // Should not happen...

        // For every good neighbour connect them in the adjacency matrix (symmetric)
        for (int nidx=0; nidx<(int)good_neighs.size(); nidx++) {
            if (SSD.corresp(good_neighs[nidx],0) == -1) continue;
            Adj((int)SSD.corresp(pIdx,0), (int)SSD.corresp(good_neighs[nidx],0)) = 1;
            Adj((int)SSD.corresp(good_neighs[nidx],0), (int)SSD.corresp(pIdx,0)) = 1;
        }
    }

    adj_before_collapse.resize(Adj.rows(), Adj.cols());
    adj_before_collapse = Adj;

    /* Edge collapse */
    // Redundant edges forming triangles are removed...
    std::vector<int> ec_neighs;
    Eigen::MatrixXd edge_rows;
    edge_rows.resize(2,3);

    while (1) {
        int tricount = 0;
        Eigen::MatrixXi skeds; // Vertice involved in triangle
        skeds.resize(0,2);
        Eigen::MatrixXd skcst;
        skcst.resize(0,1); // Lenght of triangle edges

        // For each vertex...
        for (int i=0; i<SSD.skelver.rows(); i++) {
            ec_neighs.clear();

            // ... Find connections between other vertices (Only upper triangular matrix col>i)
            for (int col=0; col<Adj.cols(); col++) {
                if (Adj(i,col) == 1 && col>i) {
                    ec_neighs.push_back(col);
                }
            }

            // Sorting redundant...
            // std::sort(ec_neighs.begin(), ec_neighs.end());

            // Determine triangles - Loops through pairs of connected vertices j and k...
            for (int j=0; j<(int)ec_neighs.size(); j++) {
                for (int k=j+1; k<(int)ec_neighs.size(); k++) {

                    // If the two vertices also are connected a triangle is found...
                    if (Adj(ec_neighs[j], ec_neighs[k]) == 1) {
                        tricount++; //triangle counter 
                        // ADD TRIANGLE EDGES 
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

        // Continue loop as long as triangles are found
        if (tricount == 0) {
            break;
        }

        // Determine smallest edge...
        Eigen::MatrixXd::Index minRow, minCol;
        skcst.minCoeff(&minRow, &minCol); // Output minimum edge length
        int idx = minRow;
        Eigen::Vector2i edge = skeds.row(idx); // get edge connections

        // Set actual vertex connections 
        edge_rows.row(0) = SSD.skelver.row(edge(0));
        edge_rows.row(1) = SSD.skelver.row(edge(1));
        SSD.skelver.row(edge(0)) = edge_rows.colwise().mean(); // First vertex set as the mean of the two 
        SSD.skelver.row(edge(1)).setConstant(std::numeric_limits<double>::quiet_NaN()); // Set second vertice invalid...

        // Update adjacency matrix
        for (int k=0; k<Adj.rows(); k++) {
            if (Adj(edge(1), k) == 1) {
                Adj(edge(0), k) = 1;
                Adj(k, edge(0)) = 1;
            }
        }
        Adj.row(edge(1)).setZero();
        Adj.col(edge(1)).setZero(); 

        // Update corresp matrix to ensure that the points assigned to removed vertex are assigned to the new merged vertex...
        for (int r=0; r<SSD.corresp.rows(); r++) {
            if (SSD.corresp(r,0) == (double)edge(1)) {
                SSD.corresp(r,0) = (double)edge(0);
            }
        }
    }
    // Final adjenceny matrix of the skeleton...
    SSD.skeladj = Adj;

    // temp_ver_cloud->clear();
    // for (int i=0; i<(int)SSD.skelver.rows(); i++) {
    //     pcl::PointXYZ ptt;
    //     ptt.x = SSD.skelver(i,0) * norm_scale + centroid(0);
    //     ptt.y = SSD.skelver(i,1) * norm_scale + centroid(1);
    //     ptt.z = SSD.skelver(i,2) * norm_scale + centroid(2);
    //     temp_ver_cloud->points.push_back(ptt);
    // }
}

void RosaMain::recenter() {
    Extra_Del ed_rr;
    std::vector<int> deleted_vertice_idxs, idxs;
    Eigen::MatrixXi ne_idxs, d_idxs;
    Eigen::MatrixXd ne_idxs_d, d_idxs_d, extract_pts, extract_nrs;
    Eigen::Vector3d proj_center, eucl_center, fuse_center;
    Eigen::MatrixXd temp_skelver, temp_skeladj_d, temp_skeladj_d2;

    // For each vertex
    for (int i=0; i<SSD.skelver.rows(); i++) {
        idxs.clear();
        for (int j=0; j<SSD.corresp.rows(); j++) {
            // Extract the indices of the points corresponding to the ith vertex
            if (SSD.corresp(j,0) == (double)i) {
                idxs.push_back(j);
            }
        }
        if (idxs.size() < 3) {
            // If a vertex has less than 3 assigned points -> Remove this vertex (will also remove previous set NaNs)
            deleted_vertice_idxs.push_back(i);
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
    int del_size = deleted_vertice_idxs.size();
    d_idxs = Eigen::Map<Eigen::MatrixXi>(deleted_vertice_idxs.data(), deleted_vertice_idxs.size(), 1);
    d_idxs_d = d_idxs.cast<double>();
    temp_skeladj_d = SSD.skeladj.cast<double>();
    Eigen::MatrixXd fill = Eigen::MatrixXd::Zero(del_size, SSD.skeladj.cols());

    temp_skelver = ed_rr.rows_del_M(d_idxs_d, SSD.skelver); // 
    temp_skeladj_d = ed_rr.rows_del_M(d_idxs_d, temp_skeladj_d);
    temp_skeladj_d2.resize(SSD.skeladj.cols(), SSD.skeladj.cols());
    temp_skeladj_d2 << temp_skeladj_d, fill;
    temp_skeladj_d2 = ed_rr.cols_del_M(d_idxs_d, temp_skeladj_d2);
    SSD.skelver = temp_skelver;
    SSD.skeladj = temp_skeladj_d2.block(0, 0, temp_skeladj_d2.cols(), temp_skeladj_d2.cols()).cast<int>();

    SSD.vertices = SSD.skelver;

    // Final edge adjustments..
    SSD.edges.resize(0,2);
    for (int i=0; i<SSD.skeladj.rows(); i++) {
        for (int j=0; j<SSD.skeladj.cols(); j++) {
            if (SSD.skeladj(i,j) == 1 && i<j) {
                SSD.edges.conservativeResize(SSD.edges.rows()+1, SSD.edges.cols());
                SSD.edges(SSD.edges.rows()-1, 0) = i;
                SSD.edges(SSD.edges.rows()-1, 1) = j;
            }
        }
    }
}

void RosaMain::restore_scale() {
    SSD.rosa_pts->clear();
    for (int i=0; i<(int)SSD.vertices.rows(); i++) {
        pcl::PointXYZ pt;
        pt.x = SSD.vertices(i,0) * norm_scale + centroid(0);
        pt.y = SSD.vertices(i,1) * norm_scale + centroid(1);
        pt.z = SSD.vertices(i,2) * norm_scale + centroid(2);
        SSD.rosa_pts->points.push_back(pt);
    }

    /* For pts after dcrosa() */
    // for (int i=0; i<(int)pset.rows(); i++) {
    //     pcl::PointXYZ pt;
    //     pt.x = pset(i,0) * norm_scale + centroid(0);
    //     pt.y = pset(i,1) * norm_scale + centroid(1);
    //     pt.z = pset(i,2) * norm_scale + centroid(2);
    //     SSD.rosa_pts->points.push_back(pt);
    // }
}

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
     // srand((unsigned)time(NULL));
     Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
     M(0,0) = v(0); 
     M(0,1) = v(1); 
     M(0,2) = v(2);
     Eigen::Vector3d new_vec, temp_vec;
 
     // Seems inefficient to just iterate until satisfaction? - Rewrite using deterministic linear algebra (cross product method)?
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

    std::vector<int> queue;
    queue.reserve(pcd_size_); // Allocate memory
    queue.emplace_back(idx); // Insert at the end of queue

    int curr;
    while (!queue.empty()) {
        curr = queue.back();
        queue.pop_back();
        isoncut[curr] = 2;
        out_indxs(out_size++, 0) = curr;

        for (size_t i = 0; i < SSD.neighs[curr].size(); ++i) {
            if (isoncut[SSD.neighs[curr][i]] == 1) {
                isoncut[SSD.neighs[curr][i]] = 3;
                queue.emplace_back(SSD.neighs[curr][i]);
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
    Eigen::MatrixXd repmat = symm_nor.transpose().replicate(num, 1); // replicate with a row-factor of num and col-factor of 1

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
    return vec;
}

int RosaMain::argmax_eigen(Eigen::MatrixXd &x) {
    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow,&maxCol);
    int idx = maxRow;
    return idx;
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr RosaMain::getRosaPoints() const {
    return SSD.rosa_pts;
}



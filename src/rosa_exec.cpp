#include "rosa_main.hpp"
#include "global_skel.hpp"

class RosaNode : public rclcpp::Node {
public:
    RosaNode() : Node("rosa_node") {
        init();
    }

    void init();
    void init_modules();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);
    void set_cloud();
    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rosa_pub_;
    rclcpp::TimerBase::SharedPtr run_timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_2_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
private:
    /* Params */
    bool init_flag = false;
    bool run_flag = false;
    int N_scans;
    
    /* Data */
    int scan_counter = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd;
    pcl::PointCloud<pcl::PointXYZ>::Ptr batch_pcd;
    geometry_msgs::msg::TransformStamped curr_tf;

    /* Utils */
    std::shared_ptr<RosaMain> skel_op;
    std::shared_ptr<GlobSkel> GS;
};

void RosaNode::init() {
    /* Get params from launch file */
    this->declare_parameter<int>("num_scans", 10);
    N_scans = this->get_parameter("num_scans").as_int();
    
    /* Initilize data structures (Before pubs/subs!) */
    current_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
    batch_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    /* ROS2 Publisher/Subscribers/Timers */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/pointcloud", 10, std::bind(&RosaNode::pcd_callback, this, std::placeholders::_1));
    rosa_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud_repub", 10);
    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&RosaNode::run, this));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debugger", 10);
    debug_pub_2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debugger_2", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rosa_dir", 10);
}

void RosaNode::init_modules() {
    /* Initialization modules and share ROS2 node */
    RCLCPP_INFO(this->get_logger(), "Initializing Modules...");
    skel_op.reset(new RosaMain);
    GS.reset(new GlobSkel);
    skel_op->init(shared_from_this());
    GS->init(shared_from_this());
    init_flag = true;
}

void RosaNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
    pcl::fromROSMsg(*pcd_msg, *current_pcd);
    *batch_pcd += *current_pcd;
    scan_counter++;
    if (scan_counter == N_scans) {
        set_cloud();
        batch_pcd->clear();
        scan_counter = 0;   
    }
}

void RosaNode::set_cloud() {
    if (!init_flag) init_modules(); // Initialize modules on first call
    if (skel_op->SSD.pts_ == nullptr) return; // Handle startup synch

    // Set transform between lidar_frame and World
    try {
        curr_tf = tf_buffer_->lookupTransform("World", "lidar_frame", tf2::TimePointZero);
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "Transform Lookup Failed: %s", ex.what());
        return;
    }
    
    // Set current pointcloud if not empty
    if (batch_pcd->empty()) return;
    pcl::copyPointCloud(*batch_pcd, *skel_op->SSD.pts_);
    run_flag = true; // Ready to run ROSA algorithm
}

void RosaNode::run() {
    if (run_flag) {
        run_flag = false;

        skel_op->main(); // Run main ROSA points algorithm
        GS->update_skel(skel_op->getRosaPoints(), curr_tf); // Update current structure skeleton... 

        RCLCPP_INFO(this->get_logger(), "ROSA Skeleton size: %ld", GS->global_skeleton->points.size());

        // Temp: Debugger publisher - Specify pointcloud quantity to visualize in Rviz2...
        sensor_msgs::msg::PointCloud2 db_out;
        // pcl::toROSMsg(*GS->global_skeleton, db_out);
        // db_out.header.frame_id = "World";
        pcl::toROSMsg(*skel_op->debug_cloud, db_out);
        db_out.header.frame_id = "lidar_frame";
        db_out.header.stamp = this->get_clock()->now();
        debug_pub_->publish(db_out);

        sensor_msgs::msg::PointCloud2 db_out_2;
        // pcl::toROSMsg(*GS->debug_cloud, db_out_2);
        // db_out_2.header.frame_id = "World";
        
        pcl::toROSMsg(*skel_op->debug_cloud_2, db_out_2);
        db_out_2.header.frame_id = "lidar_frame";
        db_out_2.header.stamp = this->get_clock()->now();
        debug_pub_2_->publish(db_out_2);
    }
}



int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RosaNode>();

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
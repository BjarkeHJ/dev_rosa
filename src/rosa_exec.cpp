#include "rosa_main.hpp"

class RosaNode : public rclcpp::Node {
public:
    RosaNode() : Node("rosa_node") {
        init();
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);
    void set_cloud();
    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rosa_pub_;
    rclcpp::TimerBase::SharedPtr run_timer_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;

private:
    /* Params */
    bool init_flag = false;
    bool run_flag = false;
    int N_scans;
    
    /* Data */
    int scan_counter = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd;
    pcl::PointCloud<pcl::PointXYZ>::Ptr batch_pcd;

    /* Utils */
    std::shared_ptr<RosaMain> skel_op;
};

void RosaNode::init() {
    /* Get params from launch file */
    this->declare_parameter<int>("num_scans", 10);
    N_scans = this->get_parameter("num_scans").as_int();


    /* Initilize data structures (Before pubs/subs!) */
    current_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
    batch_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    /* Module Initialization */
    skel_op.reset(new RosaMain);
    init_flag = true;
    
    /* ROS2 Publisher/Subscribers/Timers */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/pointcloud", 10, std::bind(&RosaNode::pcd_callback, this, std::placeholders::_1));
    rosa_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud_repub", 10);
    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&RosaNode::run, this));

    debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debugger", 10);
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
    /* Supply ROSA algorithm with current cloud */
    pcl::copyPointCloud(*batch_pcd, *skel_op->SSD.pts_);
    run_flag = true;
}

void RosaNode::run() {
    if (init_flag) {
        // Initialize RosaMain class on first iteration and pass the node...
        RCLCPP_INFO(this->get_logger(), "RosaMain initialized...");
        skel_op->init(shared_from_this());
        init_flag = false;
    }

    if (run_flag) {
        RCLCPP_INFO(this->get_logger(), "Current Cloud Size: %zu", skel_op->SSD.pts_->points.size());
        skel_op->main();
        
        // Temp: Debugger publisher - Specify pointcloud quantity to visualize in Rviz2...
        sensor_msgs::msg::PointCloud2 db_out;
        pcl::toROSMsg(*skel_op->SSD.pts_, db_out);
        db_out.header.frame_id = "Sensor";
        db_out.header.stamp = this->get_clock()->now();
        debug_pub_->publish(db_out);

        // Set flag to allow next cloud...
        run_flag = false;
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
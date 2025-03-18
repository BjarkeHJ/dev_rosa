#include "rosa_main.hpp"

class RosaNode : public rclcpp::Node {
public:
    RosaNode() : Node("rosa_node") {
        init();
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);
    void set_cloud();
    void planner();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rosa_pub_;
    rclcpp::TimerBase::SharedPtr run_timer_;

private:
    /* Params */
    int N_scans = 10;
    int scan_counter = 0;
    bool run_flag = false;

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd;
    pcl::PointCloud<pcl::PointXYZ>::Ptr batch_pcd;

    /* Utils */
    std::shared_ptr<RosaMain> skel_op;
};

void RosaNode::init() {
    /* Get params from launch file */
    this->declare_parameter<int>("rosa_node/num_scans", 10);
    N_scans = this->get_parameter("rosa_node/num_scans").as_int();
    
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/pointcloud", 10, std::bind(&RosaNode::pcd_callback, this, std::placeholders::_1));
    rosa_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud_repub", 10);
    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&RosaNode::planner, this));

    current_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
    batch_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
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

void RosaNode::planner() {
    if (run_flag) {
        RCLCPP_INFO(this->get_logger(), "Current Cloud Size: %zu", skel_op->SSD.pts_->points.size());
        skel_op->main();
        
        // Publish points
        sensor_msgs::msg::PointCloud2 rosa_out;
        pcl::toROSMsg(*skel_op->SSD.pts_, rosa_out);
        rosa_out.header.frame_id = "Sensor";
        rosa_out.header.stamp = this->get_clock()->now();
        rosa_pub_->publish(rosa_out);

        // Set flag to allow next cloud...
        run_flag = false;
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RosaNode>();

    // OBS: The node can first be parsed when it is completely constructed... Hence why the skel_op is initialized here...
    RosaMain skel_op;
    skel_op.init(node);

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
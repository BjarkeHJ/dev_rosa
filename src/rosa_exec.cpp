#include "rosa_main.hpp"

class RosaNode : public rclcpp::Node {
public:
    RosaNode() : Node("rosa_points") {
        init();
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_pub_;

private:
    int N_scans = 30;
    int scan_counter = 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd;
    pcl::PointCloud<pcl::PointXYZ>::Ptr batch_pcd;
};


void RosaNode::init() {
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/pointcloud", 10, std::bind(&RosaNode::pcd_callback, this, std::placeholders::_1));
    pcd_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud_repub", 10);

    current_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
    batch_pcd.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void RosaNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
    pcl::fromROSMsg(*pcd_msg, *current_pcd);

    *batch_pcd += *current_pcd;
    scan_counter++;    
    
    if (scan_counter == N_scans) {
        sensor_msgs::msg::PointCloud2 batch_out;
        pcl::toROSMsg(*batch_pcd, batch_out);
        batch_out.header.frame_id = pcd_msg->header.frame_id;
        batch_out.header.stamp = pcd_msg->header.stamp;
        pcd_pub_->publish(batch_out);
        batch_pcd->clear();
        scan_counter = 0;
    }
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto rosa_node = std::make_shared<RosaNode>();
    rclcpp::spin(rosa_node);
    rclcpp::shutdown();
    return 0;
}
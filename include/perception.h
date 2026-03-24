#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <general_msgs/srv/set_uint8.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

#include <string>
#include <deque>
#include "det_perception.h"
#include "seg_perception.h"
#include "cls_perception.h"
#include "multi_sub_perception.h"
#include "qr_cs_perception.h"
#include "cdt_perception.h"
#include "stereo_multi_match.h"
#include "aruco_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include "general_msgs/srv/perception_pattern.hpp"
#include "general_msgs/msg/perception_target.hpp"
#include "general_msgs/msg/download_progress.hpp"
#include "general_msgs/msg/abnormal_status.hpp"
#include "perception_common.h"
#include <map>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <json/json.h>
#include "std_msgs/msg/int32.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/convert.h>
#include <tf2/time.h>
#include <tf2/utils.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace cv;
using namespace std;

struct FusionInputs
{
    cv::Mat depth;
    cv::Mat label;
    std::vector<Detection> detections;
    cv::Mat origL; // 原始图像用于裁剪和尺寸调整
};

class PerceptionNode : public rclcpp::Node
{
public:
    PerceptionNode();
    ~PerceptionNode();

    StereoMultiMatch stereo_multi_match;
    cv::Mat P1_, P2_;
    cv::Point cs_cen_point = cv::Point(0, 0);
    float cs_max_area = 0;
    float cs_cal_angle = 0, cs_cal_dist = 0;
    
    int frq_cdt=5;
    Rect cdt_rect;
    std::vector<Detection> ct_dect_src;

private:
    int count_stereo = 0;

    std::ifstream mul_model_file;
    std::ifstream det_model_file;
    std::ifstream seg_model_file;
    std::ifstream cs_model_file;
    std::ifstream json_file;

    int left_side = 0;
    int right_side = 0;
    bool m_do_save_img_pcd = false;
    bool enable_ces_show=false;
    bool enable_height_filter_=false;  // 控制是否启用高度约束过滤
    bool enable_lawn_ratio=false;
    std::string lawn_ratio;

    int m_side_mode = 4;
    int seg_person_area = 8900;
    int det_person_area = 25000;
    bool m_side_enable_debug_show = false;
    string side_dir;
    bool m_do_save_left_img = false;
    bool m_do_save_right_img = false;

    std::string m_multi_model_name;
    std::string m_misty_model_name;
    std::string m_det_model_name;
    std::string m_seg_model_name;
    std::string m_cs_model_name;
    std::string m_sub_model_name;
    std::string m_cdt_model_name;

    double m_detection_threshold;
    float m_area_threshold;
    string m_model_infer_class_file;
    bool m_enable_front_camera;
    bool m_enable_side_camera=false;

    std::string config_path1 = "/userdata/bestmow_data/stereo_perception_model/";
    std::string config_path2 = "/app/BestMow/install/stereo_perception_multi2/share/stereo_perception_multi2/perception_conf/";

    bool m_disable_debug_camera;
    bool m_disable_front_camera;
    bool m_disable_side_camera;

    int m_side_warning_area;

    int m_worker_threads_;
    int m_queue_size_;
    std::vector<std::thread> workers_;

    int m_fusion_model;
    int temp_fusion_model;
    int temp_erode_pixel;
    int m_erode_pixel;
    string m_filename;
    map<int, string> mode_way;
    string side_time;
    string side_func;

    multi_perception multiPerception;
    multi_perception mulSubPerception;
    seg_perception segPerception;
    qr_cs_perception qrCsPerception;
    det_perception detPerception;
    ArucoDetector m_aruco_detector;
    det_cable_perception cdtPerception;

    bool _do_detection;
    map<int, string> det_map_class;

    struct stereoImg
    {
        Mat stereo_left;
        Mat stereo_right;
        rclcpp::Time stamp; // builtin_interfaces::msg::Time
    } stereo_img_;

    stereoImg select_stereo;

    mutex mtxStereo, mtxDepth, mtxMul;           // 各队列的互斥锁
    condition_variable cvStereo, cvDepth, cvMul; // 条件变量

    int frame_counter = 0; // 帧计数器
    int skip_frames = 0;

    bool has_images_ = false;
    bool has_rain_dect = false;

    int ret_rain = 5;

    double fx_, fy_, cx_, cy_, baseline_;
    bool left_camera_info_received_;
    bool right_camera_info_received_;

    void stereoImgNumProducetimerCallBack(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &left,
                                          const sensor_msgs::msg::CompressedImage::ConstSharedPtr &right);
    double processImg(cv::Mat &ori_mat, const cv::Mat &img_label, uchar label, cv::Mat &dst_mask);
    bool is_compost_hsv(Vec3b &hsv_pixel);
    string initializeFromFile(const std::string &path1, const std::string &path2);

    bool readModelVersion(const std::string &filePath, std::string &versionOut);
    std::string getVersionedPath(const std::string &pathA, const std::string &pathB);

    rclcpp::SubscriptionOptions reen_options;
    rclcpp::SubscriptionOptions mutu_options;

    std::thread stereo_processing_thread_;
    rclcpp::CallbackGroup::SharedPtr reen_callback_group_;
    rclcpp::CallbackGroup::SharedPtr mutu_callback_group_;
    rclcpp::TimerBase::SharedPtr pub_timer_;

    std::mutex stereo_queue_mutex_;                  // 用于保护队列的互斥锁
    std::condition_variable stereo_queue_condition_; // 用于线程同步
    bool stereo_stop_processing_ = false;

    std::atomic<bool> is_processing_;

    std::mutex image_lock_;
    std::atomic<bool> new_left_;
    std::atomic<bool> new_right_;
    std::atomic<bool> enable_point_cloud_localization_;

    rclcpp::CallbackGroup::SharedPtr left_cb_group_;
    rclcpp::CallbackGroup::SharedPtr right_cb_group_;

    std::queue<stereoImg> stereo_queue_;
    rclcpp::TimerBase::SharedPtr det_left_pub_timer_;
    rclcpp::TimerBase::SharedPtr det_right_pub_timer_;

    std::queue<Mat> side_queue_;
    std::mutex side_queue_mutex_;                  // 用于保护队列的互斥锁
    std::condition_variable side_queue_condition_; // 用于线程同步
    bool side_stop_processing_ = false;

    // 处理线程
    std::thread side_left_processing_thread_;
    std::thread side_right_processing_thread_;
    std::thread side_processing_thread_;

    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};

    rclcpp::TimerBase::SharedPtr process_timer_;

    void node_param_init();
    void stereo_process();
    void sideImageCallback();

    void side_left_process();
    void side_right_process();
    void det_side_left_process();
    void det_side_right_process();

    void det_side_process();
    void save_img_pcd(Mat imageL, Mat imageR, pcl::PointCloud<pcl::PointXYZL> xyzi_cloud);
    void save_img_rgb_pcd(Mat imageL, Mat imageR, pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbi_cloud);
    void save_img(Mat &imageL, Mat &imageR, string name);

    void leftCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void rightCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void updateCameraParameters();

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage> MySyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> stereo_left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> stereo_right_sub_;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr left_sub_, right_sub_;
    sensor_msgs::msg::CompressedImage::ConstSharedPtr last_left_, last_right_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr stereo_left_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr stereo_right_info_sub_;

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr color_depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_points_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void stereoImgtimerCallBack(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &last_left_,
                                const sensor_msgs::msg::CompressedImage::ConstSharedPtr &last_right_);

    void stereoImgProducetimerCallBack(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &left,
                                       const sensor_msgs::msg::CompressedImage::ConstSharedPtr &right);

    void handleLocalization(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    void processImages();

    void thread_depth(const struct stereoImg &select_stereo, std::shared_ptr<FusionInputs> fusion_inputs);
    void thread_det_seg(const struct stereoImg &select_stereo, std::shared_ptr<FusionInputs> fusion_inputs);
    void thread_fusion(const struct stereoImg &select_stereo, FusionInputs fusion_inputs);

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::mutex depth_mutex_, result_mutex_;
    bool stop_threads_ = false;

    std::vector<std::thread> worker_threads_;

    void leftImgTimeCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void rightImgTimeCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>> left_img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>> right_img_sub_;

    rclcpp::TimerBase::SharedPtr ring_timer_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr seg_pub_;

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr side_pub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr side_left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr side_right_sub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr side_left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr side_right_pub_;

    rclcpp::Publisher<general_msgs::msg::PerceptionTarget>::SharedPtr left_side_det_info_publisher_;
    rclcpp::Publisher<general_msgs::msg::PerceptionTarget>::SharedPtr right_side_det_info_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr left_side_det_boxes_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr right_side_det_boxes_publisher_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_cam_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr seg_image_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_{nullptr};

    Eigen::Matrix3d rot_base2camera_;
    Eigen::Vector3d t_base2camera_in_camera_;
    Eigen::Vector3d t_base2aruco_;
    Eigen::Vector3d t_aruco2base_;

    Eigen::Quaternion<double> q_base2camera_;
    Eigen::Quaternion<double> q_base2aruco_;
    Eigen::Quaternion<double> q_aruco2base_;

    bool tf_received_;

    bool receive_TF_FC2Base();
    void caculateArucoPose(const Mat &seg_lab, const Mat &depth_img, const rclcpp::Time timestamp);
    void caculateOpencvArucoPose(const std::vector<cv::Point2f> &outPoly, const Mat &depth_img, Mat left_img, const rclcpp::Time timestamp);
    bool processArucoPointCloud(const Mat &dyna_mat, const Mat &depth_img, Eigen::Quaterniond &q_out,
                                Eigen::Vector3d &t_out, const rclcpp::Time timestamp);
    bool processOpencvArucoPointCloud(const std::vector<cv::Point2f> &PolyF, const Mat &depth_img, Mat &left_img, Eigen::Quaterniond &q_out,
                                      Eigen::Vector3d &t_out, const rclcpp::Time timestamp);

    void publishAruco2Camera(Eigen::Quaternion<double> &q_cam, Eigen::Vector3d &t_cam, const rclcpp::Time timestamp);
    void publishBase2Aruco(Eigen::Quaternion<double> &q_base2aruco, Eigen::Vector3d &t_base2aruco, const rclcpp::Time timestamp);
    void publishOdomBase2Aruco(Eigen::Quaternion<double> &q_base2aruco, Eigen::Vector3d &t_base2aruco, const rclcpp::Time timestamp);
    void broadcastTF_aruco2Base(Eigen::Quaternion<double> &q_aruco2base, Eigen::Vector3d &t_aruco2base, const rclcpp::Time timestamp);

    std::vector<unsigned char> det_compressed_data;
    std::vector<int> det_compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};
    sensor_msgs::msg::CompressedImage det_compressed_msg;

    std::vector<unsigned char> seg_compressed_data;
    std::vector<int> seg_compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};
    sensor_msgs::msg::CompressedImage seg_compressed_msg;

    std::vector<unsigned char> side_left_compressed_data;
    std::vector<unsigned char> side_right_compressed_data;

    std::vector<unsigned char> side_left_seg_compressed_data;
    std::vector<unsigned char> side_right_seg_compressed_data;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_img_pcd_server_;
    rclcpp::Service<general_msgs::srv::PerceptionPattern>::SharedPtr service_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_localization_server_;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr dynamic_publisher_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr point_size_pub_;

    void handleService(const std::shared_ptr<general_msgs::srv::PerceptionPattern::Request> request,
                       const std::shared_ptr<general_msgs::srv::PerceptionPattern::Response> response);

    bool is_detectd_pedestrian(const std::vector<Detection> &dect_src, const cv::Mat img_src, string side, int count);

    void angle_distance(Mat &depth, cv::Point &cs_c_point, float &cs_angle, float &cs_dist);

    void publish_rgb_point_cloud(const rclcpp::Time &xyz_rgbl_stamp, pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud);

    rclcpp::Subscription<general_msgs::msg::DownloadProgress>::SharedPtr ota_download_subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr ota_download_publisher_;
    void download_progress_callback(const general_msgs::msg::DownloadProgress::SharedPtr msg);

    rclcpp::Publisher<general_msgs::msg::AbnormalStatus>::SharedPtr perception_status_publisher_;
    void perception_status_callback(const general_msgs::msg::AbnormalStatus::SharedPtr msg);

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr cable_tie_publisher_;
    int init_model(std::string, std::string);

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr camera_freq_sub_;
    void camera_freq_callback(const std_msgs::msg::Int32::SharedPtr msg);

    void leftCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void rightCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    double secnano(const builtin_interfaces::msg::Time &t);

    void publishStatus(int err_code, const std::string& description, bool update_state = true);

    void startStatusTimer();

    rclcpp::TimerBase::SharedPtr status_timer_;
    general_msgs::msg::AbnormalStatus abnor_msg = general_msgs::msg::AbnormalStatus();

    std::atomic<int> current_err_code_{150};
    std::atomic<bool> image_input_valid_{false};
    std::atomic<int64_t> last_process_time_ns_{0};
    std::atomic<bool> status_timer_started_{false};

    std::condition_variable image_cv_;

    rclcpp::SubscriptionOptions left_opts;
    rclcpp::SubscriptionOptions right_opts;
};

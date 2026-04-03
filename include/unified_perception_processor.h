#ifndef UNIFIED_PERCEPTION_PROCESSOR_H
#define UNIFIED_PERCEPTION_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <memory>
#include "perception.h"

// Forward declarations for perception modules
class cdt_perception;
class det_perception;
class seg_perception;
class qr_cs_perception;
class multi_perception;
class dsg_perception;
class StereoMultiMatch;
class stereo_point_cloud;

namespace offline_perception {

// Use original class names for simplicity and consistency with existing code
using CdtPerception = cdt_perception;
using DetPerception = det_perception;
using SegPerception = seg_perception;
using QrCsPerception = qr_cs_perception;
using MultiPerception = multi_perception;
using MulSubPerception = multi_perception; // MulSub often uses same class
using DsgPerception = dsg_perception;
using DsgMultiPerception = dsg_perception;
using StereoPointCloudRGBL = stereo_point_cloud;

struct Config {
    int infer_mode = 5;
    int image_width = 640;
    int image_height = 384;
    int erode_pixel = 205;
    float area_threshold = 0.5;
    float detection_threshold = 0.51;
    bool enable_height_filter = false;
    bool enable_cdt = true;
    bool enable_debug_show = true;

    std::string model_dir = "../models/";
    std::string cdt_model = "cdt_20251125_640x384.bin";
    std::string det_model = "det_20241106_640x480.bin";
    std::string seg_model = "seg_20250421_640x384.bin";
    std::string multi_model = "mul_20250918_640x384.bin";
    std::string cs_model = "cqr_20250821_640x384_yolov8n.bin";
    std::string sub_model = "sub_20260303_640x384.bin";
    std::string dsg_model = "dsg_20260115_640x384.bin";
    std::string dsg_multi_model = "dsg_multi_20260401_640x384.bin";

    // Visualisation paths
    std::string finalPicDir = "./output/images/";
    std::string finalPcdDir = "./output/pcd/";
};

struct ProcessResult {
    pcl::PointCloud<pcl::PointXYZRGBL> point_cloud;
    cv::Mat depth_map;
    cv::Mat label_map;
    cv::Mat visualization;
    double process_time_ms = 0.0;
    int frame_id = 0;
    std::string mode_name;
};

class UnifiedPerceptionProcessor {
public:
    explicit UnifiedPerceptionProcessor(const Config& cfg);
    ~UnifiedPerceptionProcessor();

    bool init();
    ProcessResult process(const cv::Mat& left_img, const cv::Mat& right_img, int frame_id, const std::string& image_name = "");

private:
    Config config_;

    // Pointers to individual perception modules (encapsulated)
    std::unique_ptr<StereoMultiMatch> stereo_matcher_;
    std::unique_ptr<CdtPerception> cdt_perceptor_;
    std::unique_ptr<DetPerception> det_perceptor_;
    std::unique_ptr<SegPerception> seg_perceptor_;
    std::unique_ptr<QrCsPerception> qr_cs_perceptor_;
    std::unique_ptr<MultiPerception> multi_perceptor_;
    std::unique_ptr<MulSubPerception> sub_perceptor_;
    std::unique_ptr<DsgPerception> dsg_perceptor_;
    std::unique_ptr<DsgMultiPerception> dsg_multi_perceptor_;

    // Utility functions
    static float computeStraightEdgeScore(const cv::Mat& grayROI, const cv::Mat& compMask);
    void refineObstacleByColorAndEdge(cv::Mat& label, const cv::Mat& bgrImg,
                                       int min_area = 200,
                                       float min_edge_score = 0.20f,
                                       float min_solidity = 0.50f,
                                       float min_green_ratio = 0.50f);

    // DSG specific helpers
    static void applyDarknessFilter(cv::Mat &label, const cv::Mat &bgrImg, int darkness_threshold = 30);
    static void filterNighttimeDepth(cv::Mat &depth, const cv::Mat &label_map);
    static void filterNighttimePointCloud(pcl::PointCloud<pcl::PointXYZRGBL> &cloud, const cv::Mat &label_map);
    static void processLabelContours(cv::Mat &label_img, bool &all_neighbors_are_lawn);

    // Integration and Visualization helpers
    void filterLabelDect(cv::Mat& src_lab, std::vector<Detection>& dect_src,
                        cv::Mat& lab_dst, std::vector<Detection>& dect_dst,
                        bool enable_det = true);
    cv::Mat drawResultOptimized(const cv::Mat& img, const cv::Mat& label,
                               const std::vector<Detection>& dect, cv::Mat& img_seg_show, bool enable_draw_box = false);
    cv::Rect get_cdt_rect(const std::vector<Detection>& detections);

    // Mode-specific processing functions
    ProcessResult processMode0(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode1(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode2(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode3(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode4(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode5(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode6(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode7(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
    ProcessResult processMode8(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR);
};

} // namespace offline_perception

#endif // UNIFIED_PERCEPTION_PROCESSOR_H

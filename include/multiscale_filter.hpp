#ifndef MULTISCALE_FILTER_HPP_
#define MULTISCALE_FILTER_HPP_

#include <opencv2/opencv.hpp>
#include <queue>
#include <unordered_set>
#include <perception_common.h>

static bool checkLabelInRegion(const cv::Mat& img_label, Bbox &obj_box) __attribute__((unused));
static bool checkLabelInRegion(const cv::Mat& img_label, Bbox &obj_box) {
//    Bbox smbox= det.bbox;
    int xmin = obj_box.xmin;
    int xmax = obj_box.xmax;
    int ymin = obj_box.ymin;
    int ymax = obj_box.ymax;

    // 遍历指定区域内的每个像素
    for (int h = ymin; h <= ymax; h+=2) {
        for (int w = xmin; w <= xmax; w+=2) {
            int result_id = img_label.at<uint8_t>(h, w);
//            if (result_id == 2 || result_id==3) {
//                return false;
//            }
            if (result_id == 1 || result_id==4  || result_id ==5) {
                return false;
            }
        }
    }
    return true;
}

// Custom hash function for cv::Point
struct PointHash {
    std::size_t operator()(const cv::Point& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
};

struct MultiScaleFilterParams {
    float diff_threshold;
    float potential_threshold;
    int growing_window_size;
    float growing_threshold;
    float excessive_threshold;

    MultiScaleFilterParams() {};

    MultiScaleFilterParams(float diff_th, float pot_th,
        int win_size, float grow_th,
        float excess_th)
        : diff_threshold(diff_th), potential_threshold(pot_th),
        growing_window_size(win_size), growing_threshold(grow_th),
        excessive_threshold(excess_th) {
    }
};

std::pair<std::vector<cv::Point>, cv::Mat>
bfsMarkConnected(const cv::Mat& disparity, int start_y, int start_x,
    cv::Mat& visited, float max_diff);

cv::Mat smallRegionHighDisparityFilter(const cv::Mat& disparity,
    float disp_threshold,
    int area_threshold, 
    cv::Mat obstacle_mask = cv::Mat());

cv::Mat detectDisparityAnomalies(const cv::Mat& reference_disparity,
    const cv::Mat& target_disparity,
    const MultiScaleFilterParams& params,
    const cv::Mat& gray);

cv::Mat detectDisparityAnomaliesOptimized(const cv::Mat& reference_disparity,
    const cv::Mat& target_disparity,
    const MultiScaleFilterParams& params,
    const cv::Mat& gray, 
    const cv::Mat obstacle_mask = cv::Mat());

bool isLowTextureArea(const std::vector<cv::Point>& connected_points, const cv::Mat& image);
//bool isLowTextureArea(const cv::Mat& mask, const cv::Mat& image);
bool isLowTextureArea(const cv::Rect& roi, const cv::Mat& image);

cv::Mat computeMultiScaleDisparity(cv::Ptr<cv::StereoBM> half_top_stereo_block_matcher,
    cv::Ptr<cv::StereoSGBM> half_bottom_stereo_block_matcher,
    const cv::Mat& gray_left, const cv::Mat& gray_right,
    cv::Mat& disparity, const MultiScaleFilterParams& orig_param, const MultiScaleFilterParams& half_param,
    cv::Mat obstacle_mask = cv::Mat(),
    double fx = 244.9567633, double fy = 244.9567633,
    double cx = 321.05016538, double cy = 234.44410892,
    double baseline = 0.0799,
    bool enable_height_filter = true);

#endif
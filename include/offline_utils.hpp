#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "detection_type.hpp" // Detection, Bbox

// 通用的离线调试辅助工具：点云三视图、标签统计、标签后处理等。

class PointCloudVisualizer {
public:
  static void saveThreeViews(const pcl::PointCloud<pcl::PointXYZRGBL> &cloud,
                             const std::string &output_prefix);

  static void
  saveLabelDistribution(const pcl::PointCloud<pcl::PointXYZRGBL> &cloud);
};

// 标签图分布统计
void printLabelDistribution(const cv::Mat &img_label);
std::map<int, int> getLabelDistribution(const cv::Mat &img_label);

// 多任务 / 子任务标签 + 检测结果融合
void filterLabelDect(cv::Mat &src_lab, std::vector<Detection> &dect_src,
                     cv::Mat &lab_dst, std::vector<Detection> &dect_dst,
                     bool enable_det = true);

cv::Mat processLabelsOptimizedPipeline(const cv::Mat &label_img,
                                       const cv::Mat &rgb_img, float threshold);

// CDT 矩形区域
cv::Rect get_cdt_rect(const std::vector<Detection> &ct_dect_src);

// ===================== 可视化 & 结果输出处理 =====================
void convertIdToRGB(const cv::Mat &img_lab, cv::Mat &parsing_img);
cv::Mat drawResult(cv::Mat &img_src, cv::Mat &img_lab,
                   std::vector<Detection> &dect_src, cv::Mat &img_seg_show);

void convertIdToRGBOptimized(const cv::Mat &img_lab, cv::Mat &parsing_img);
cv::Mat drawResultOptimized(cv::Mat &img_src, cv::Mat &img_lab,
                            std::vector<Detection> &dect_src,
                            cv::Mat &img_seg_show);

// 保存带 RGB+label 的 PCD
int savePcdfile_with_rgb_label(
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbi_cloud, std::string save_name);

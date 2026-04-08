/**
 * @file offline_perception_debug.cpp
 * @brief 离线感知调试工具 - 按照ROS2 mode (0-6) 逻辑实现
 * @description 支持多种感知模式：depth-only, detection, segmentation,
 * multi-task等
 */

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

// 引入感知模块头文件
#include "cdt_perception.h"
#include "det_perception.h"
#include "multi_sub_perception.h"
#include "offline_utils.hpp"
#include "qr_cs_perception.h"
#include "seg_perception.h"
#include "stereo_multi_match.h"
#include "stereo_point_cloud_rgbl.h"
#include "dsg_perception.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

map<int, string> mul_map_class;

void initMulClassMap()
{
  mul_map_class[0] = "unla";
  mul_map_class[1] = "back";
  mul_map_class[2] = "gras";
  mul_map_class[3] = "road";
  mul_map_class[4] = "dyna";
  mul_map_class[5] = "stat";
  mul_map_class[6] = "wall";
  mul_map_class[7] = "vehi";
  mul_map_class[8] = "pole";
  mul_map_class[9] = "impa";
  mul_map_class[10] = "depr";
  mul_map_class[11] = "bush";
  mul_map_class[12] = "limb";

  mul_map_class[100] = "pole";
  mul_map_class[101] = "obst";
  mul_map_class[102] = "fixo";
  mul_map_class[103] = "car";
  mul_map_class[104] = "stat";
  mul_map_class[105] = "dyna";
  mul_map_class[106] = "chst";
  mul_map_class[107] = "pers";
}

// 建议将查找表定义为全局或类的静态成员，避免重复构造
static const std::array<cv::Vec3b, 256> getColorLookupTable()
{
  std::array<cv::Vec3b, 256> lut;
  lut.fill(cv::Vec3b(0, 0, 0)); // 默认黑色

  // 初始化特定 ID 的颜色 (BGR 顺序)
  lut[0] = cv::Vec3b(0, 0, 0);        // black
  lut[1] = cv::Vec3b(200, 0, 0);      // background
  lut[2] = cv::Vec3b(102, 255, 100);  // grass
  lut[3] = cv::Vec3b(0, 89, 118);     // road
  lut[4] = cv::Vec3b(0, 255, 255);    // dynamic
  lut[5] = cv::Vec3b(0, 0, 255);      // static_obstacle
  lut[6] = cv::Vec3b(0, 165, 255);    // wall
  lut[7] = cv::Vec3b(147, 20, 255);   // vehicle
  lut[8] = cv::Vec3b(255, 255, 0);    // pole
  lut[9] = cv::Vec3b(48, 130, 245);   // impassable
  lut[10] = cv::Vec3b(128, 64, 0);    // depression
  lut[11] = cv::Vec3b(34, 139, 34);   // bush
  lut[12] = cv::Vec3b(203, 192, 255); // limb_bush
  lut[13] = cv::Vec3b(226, 43, 138);  // CES_arod

  // 100+ ID 映射
  lut[100] = cv::Vec3b(0, 0, 255);   // pole
  lut[101] = cv::Vec3b(0, 0, 255);   // obst
  lut[102] = cv::Vec3b(0, 0, 255);   // fixo
  lut[103] = cv::Vec3b(255, 0, 255); // car
  lut[104] = cv::Vec3b(0, 0, 255);   // stat
  lut[105] = cv::Vec3b(0, 255, 255); // dyna
  lut[106] = cv::Vec3b(255, 255, 0); // charge_station

  return lut;
}


// 统计单通道 dst_label 中不同 ID 的数量
// void countLabelStatistics(const cv::Mat &dst_label, const std::string &image_name)
void countLabelStatistics(const cv::Mat &dst_label)
{
  // if (dst_label.empty())
  // {
  //   std::cout << "[Label Statistics] Empty label image: " << image_name << std::endl;
  //   return;
  // }

  std::map<int, int> id_counts;
  int total_pixels = dst_label.total();

  // 统计每个 ID 的像素数量
  for (int i = 0; i < dst_label.rows; ++i)
  {
    const uchar *row_ptr = dst_label.ptr<uchar>(i);
    for (int j = 0; j < dst_label.cols; ++j)
    {
      int id = static_cast<int>(row_ptr[j]);
      id_counts[id]++;
    }
  }

  // 输出统计结果
  // std::cout << "\n========== Label Statistics for " << image_name << " ==========" << std::endl;
  std::cout << "Image size: " << dst_label.cols << " x " << dst_label.rows
            << " (Total: " << total_pixels << " pixels)" << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;
  std::cout << std::setw(10) << "ID"
            << std::setw(15) << "Pixel Count"
            << std::setw(12) << "Percentage" << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;

  // 按 ID 排序输出
  for (const auto &pair : id_counts)
  {
    int id = pair.first;
    int count = pair.second;
    double percentage = (static_cast<double>(count) / total_pixels) * 100.0;

    std::cout << std::setw(10) << id
              << std::setw(15) << count
              << std::setw(11) << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
  }
  std::cout << "===================================================" << std::endl;
}


// 离线感知处理器 - 支持多种模式
class OfflinePerceptionProcessor
{
public:
  // 配置参数
  struct Config
  {
    int infer_mode = 7;               // 推理模式 0-6
    // int erode_pixel = 205;            // 腐蚀像素
    int erode_pixel = 205;            // 腐蚀像素
    float area_threshold = 0.5;       // 区域阈值
    float detection_threshold = 0.3; // 检测阈值
    float m_area_threshold = 0.5;
    bool enable_height_filter_ = false;
    bool enabel_cdt = false;
    int frq_cdt = 5;
    bool m_enable_debug_show = true;

    // 颜色后处理相关参数
    bool enable_red_brick_refine_ = false;  // 是否启用红色砖头颜色后处理
    int red_brick_min_area_ = 200;          // 红色砖头最小面积阈值

    // 深度补全策略: 0=禁用, 1=仅检测框, 2=仅语义, 3=两者都用
    int depth_inpainting_strategy_ = 0;

    string finalPicDir = ""; // 最终图片输出目录
    string finalPcdDir = ""; // 最终PCD输出目录

    string cdt_model = "../models/cdt_20251125_640x384.bin";
    string det_model = "../models/det_20241106_640x480.bin";
    string seg_model = "../models/seg_20250421_640x384.bin";
    string multi_model = "../models/mul_20250918_640x384.bin";
    string cs_model = "../models/cqr_20250821_640x384_yolov8n.bin";
    // string sub_model = "../models/sub_20260105_640x384.bin";
    // string sub_model = "../models/sub_20260225_640x384.bin";
    string sub_model = "../models/sub_20260303_640x384.bin";
    string dsg_multi_model = "../models/dsg_multi_20260403_640x384.bin";
  };

  struct ProcessResult
  {
    pcl::PointCloud<pcl::PointXYZRGBL> point_cloud;
    Mat depth_map;
    Mat label_map;
    cv::Mat visualization;
    double process_time_ms;
    int frame_id;
    string mode_name;
  };

  OfflinePerceptionProcessor(const Config &cfg) : config_(cfg) {}

  bool init()
  {
    cout << "\n========== Initializing Perception Modules ==========" << endl;

    // 初始化立体匹配
    stereo_multi_match.stereo_multi_param_init();
    cout << "[✓] Stereo matcher initialized" << endl;

    cdtPerception_.perception_init(config_.cdt_model.c_str());
    cout << "[✓] Cdt-task model initialized: " << config_.cdt_model << endl;

    // 根据模式初始化对应的感知模块
    if (config_.infer_mode == 0)
    {
      cout << "[Mode 0] Disable mode - no processing" << endl;
    }
    else if (config_.infer_mode == 1)
    {
      cout << "[Mode 1] Only depth mode - no DL models needed" << endl;
    }
    else if (config_.infer_mode == 2)
    {
      detPerception_.perception_init(config_.det_model.c_str(),
                                     config_.detection_threshold);
      cout << "[✓] Detection model initialized: " << config_.det_model << endl;
    }
    else if (config_.infer_mode == 3)
    {
      segPerception_.perception_init(config_.seg_model.c_str());
      cout << "[✓] Segmentation model initialized: " << config_.seg_model
           << endl;
    }
    else if (config_.infer_mode == 4)
    {
      qrCsPerception_.perception_init(config_.cs_model.c_str());
      cout << "[✓] CS/QR model initialized: " << config_.cs_model << endl;
    }
    else if (config_.infer_mode == 5)
    {
      multiPerception_.perception_init(config_.multi_model.c_str());
      cout << "[✓] Multi-task model initialized: " << config_.multi_model
           << endl;
    }
    else if (config_.infer_mode == 6)
    {
      mulSubPerception.perception_init(config_.sub_model.c_str());
      cout << "[✓] Sub-task model initialized: " << config_.sub_model << endl;
    }
    else if (config_.infer_mode == 7)
    {
      dsgPerception_.perception_init(config_.dsg_multi_model.c_str());
      cout << "[✓] DSG Night model initialized: " << config_.dsg_multi_model << endl;
    }

    cout << "===================================================\n"
         << endl;
    return true;
  }

  ProcessResult process(const Mat &left_img, const Mat &right_img, int frame_id,
                        const string &image_name = "")
  {
    auto start_time = chrono::high_resolution_clock::now();

    ProcessResult result;
    result.frame_id = frame_id;
    current_frame_id_ = frame_id;     // 保存当前帧ID
    current_image_name_ = image_name; // 保存当前图像文件名

    if (left_img.empty())
    {
      cerr << "[Error] process() received empty left image for frame "
           << frame_id << ", image name: " << image_name << endl;
      return result;
    }

    // 转换为灰度图用于立体匹配（右图可能为空）
    Mat grayImageLeft, grayImageRight;
    cvtColor(left_img, grayImageLeft, COLOR_BGR2GRAY);
    if (!right_img.empty())
    {
      cvtColor(right_img, grayImageRight, COLOR_BGR2GRAY);
    }

    // // Resize到640x384 (根据模型输入要求)
    // Mat resized_left, resized_right;
    // resize(left_img, resized_left, Size(640, 384));
    // resize(right_img, resized_right, Size(640, 384));
    // resize(grayImageLeft, grayImageLeft, Size(640, 384));
    // resize(grayImageRight, grayImageRight, Size(640, 384));

    cout << "\n======= Processing Frame " << frame_id << " =======" << endl;
    cout << "Mode: " << config_.infer_mode << endl;

    // 根据模式调用不同的处理函数
    switch (config_.infer_mode)
    {
    case 0:
      result =
          processMode0(left_img, grayImageRight, grayImageLeft, grayImageRight);
      break;
    case 1:
      result =
          processMode1(left_img, grayImageRight, grayImageLeft, grayImageRight);
      break;
    case 2:
      result =
          processMode2(left_img, grayImageRight, grayImageLeft, grayImageRight);
      break;
    case 3:
      result =
          processMode3(left_img, grayImageRight, grayImageLeft, grayImageRight);
      break;
    case 4:
      result =
          processMode4(left_img, grayImageRight, grayImageLeft, grayImageRight);
      break;
    case 5:
      result = processMode5(left_img, right_img, grayImageLeft, grayImageRight);
      break;
    case 6:
      result = processMode6(left_img, right_img, grayImageLeft, grayImageRight);
      break;
    case 7:
      result = processMode7(left_img, right_img, grayImageLeft, grayImageRight);
      break;
    default:
      cerr << "[Error] Invalid mode: " << config_.infer_mode << endl;
      break;
    }

    auto end_time = chrono::high_resolution_clock::now();
    result.process_time_ms =
        chrono::duration<double, milli>(end_time - start_time).count();
    result.frame_id = frame_id;

    cout << "Total processing time: " << fixed << setprecision(2)
         << result.process_time_ms << " ms" << endl;
    cout << "Point cloud size: " << result.point_cloud.size() << " points"
         << endl;

    return result;
  }

private:
  Config config_;
  StereoMultiMatch stereo_multi_match;
  cdt_perception cdtPerception_;
  det_perception detPerception_;
  seg_perception segPerception_;
  multi_perception multiPerception_;
  qr_cs_perception qrCsPerception_;
  multi_perception mulSubPerception;
  dsg_perception dsgPerception_;
  int current_frame_id_ = 0;       // 当前处理的帧ID
  string current_image_name_ = ""; // 当前处理的图像文件名（不含扩展名）

  /**
   * @brief 处理标签轮廓 (DSG 逻辑)
   */
  void processLabelContours(cv::Mat& label_img, bool& all_neighbors_are_lawn)
  {
    if (label_img.empty())
      return;

    if (cv::countNonZero(label_img == 2) == 0)
      return;

    cv::Mat result = label_img.clone();
    cv::Mat mask_non_target = (label_img != 0) & (label_img != 1) & (label_img != 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_non_target, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (const auto& contour : contours)
    {
      cv::Mat region_mask = cv::Mat::zeros(label_img.size(), CV_8UC1);
      cv::fillPoly(region_mask, {contour}, cv::Scalar(255));

      int lawn_neighbor_count = 0;
      int label1_neighbor_count = 0;
      int other_neighbor_count = 0;
      int total_neighbor_count = 0;
      bool region_touches_boundary = false;

      for (const auto& point : contour)
      {
        int x = point.x;
        int y = point.y;

        if (x == 0 || x == label_img.cols - 1 || y == 0 || y == label_img.rows - 1)
          region_touches_boundary = true;

        if (x < 1 || x >= label_img.cols - 1 || y < 1 || y >= label_img.rows - 1)
          continue;

        for (int i = 0; i < 8; i++)
        {
          int nx = x + dx[i];
          int ny = y + dy[i];

          if (nx >= 0 && nx < label_img.cols && ny >= 0 && ny < label_img.rows)
          {
            if (region_mask.at<uchar>(ny, nx) == 0)
            {
              total_neighbor_count++;
              uchar neighbor_label = label_img.at<uchar>(ny, nx);
              if (neighbor_label == 2)
                lawn_neighbor_count++;
              else if (neighbor_label == 1)
                label1_neighbor_count++;
              else
                other_neighbor_count++;
            }
          }
        }
      }

      float lawn_ratio = (total_neighbor_count > 0) ? (float)lawn_neighbor_count / total_neighbor_count : 0.0f;
      bool should_mark_as_13 = (label1_neighbor_count == 0) && (lawn_ratio >= 0.8f || region_touches_boundary);

      if (should_mark_as_13)
      {
        for (int y = 0; y < label_img.rows; y++)
        {
          for (int x = 0; x < label_img.cols; x++)
          {
            if (region_mask.at<uchar>(y, x) == 255)
            {
              if (label_img.at<uchar>(y, x) != 0 && label_img.at<uchar>(y, x) != 1 && label_img.at<uchar>(y, x) != 2)
                result.at<uchar>(y, x) = 13;
            }
          }
        }
        all_neighbors_are_lawn = true;
      }
    }
    label_img = result;
  }

  /**
   * @brief 计算直线边缘评分
   *
   * 使用Canny边缘检测和霍夫直线检测来评估区域内边缘的直线程度。
   * 返回值为直线边缘像素占总边缘像素的比例。
   *
   * @param grayROI 灰度图ROI
   * @param compMask 连通域掩码
   * @return 直线边缘评分 [0, 1]
   */
  static float computeStraightEdgeScore(const cv::Mat &grayROI,
                                         const cv::Mat &compMask)
  {
    cv::Mat edges;
    cv::Canny(grayROI, edges, 50, 150);
    edges &= compMask;

    int total_edge_pixels = cv::countNonZero(edges);
    if (total_edge_pixels < 10)
      return 0.0f;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 15, 10, 5);
    if (lines.empty())
      return 0.0f;

    cv::Mat line_mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    for (const auto &l : lines)
    {
      cv::line(line_mask, cv::Point(l[0], l[1]),
               cv::Point(l[2], l[3]), 255, 2);
    }
    cv::Mat line_edge = line_mask & edges;
    int line_edge_pixels = cv::countNonZero(line_edge);

    return static_cast<float>(line_edge_pixels) /
           static_cast<float>(total_edge_pixels);
  }

  /**
   * @brief 基于颜色+边缘+形状的非植被障碍物后处理 (含绿色前置条件)
   *
   * 流程:
   *   0. 绿色前置条件: ROI 内严格绿色 H∈[25,85] 占比 >= min_green_ratio 才继续
   *      (黄草场景自动跳过, 避免大量误检)
   *   1. 非植被色筛选: H 不在 [15, 85] 范围
   *   2. 语义约束: 只处理 label==2(草坪) 或 label==3(道路) 的像素
   *   3. ROI 垂直约束: 只处理图像 50%~90% 高度范围
   *   4. 形态学去噪 + 连通域分析
   *   5. 逐区域评估: 直线边缘评分 + 凸度 → 判定障碍物
   *
   * @param label         分割标签图 (CV_8UC1), 就地修改
   * @param bgrImg        BGR 彩色图 (与 label 同尺寸)
   * @param min_area      最小连通域面积 (像素)
   * @param min_edge_score 最小直线边缘评分 [0, 1]
   * @param min_solidity  最小凸度 [0, 1]
   * @param min_green_ratio ROI 内严格绿色最低占比 [0, 1]
   */
  void refineObstacleByColorAndEdge(cv::Mat &label, const cv::Mat &bgrImg,
                                     int min_area = 200,
                                     float min_edge_score = 0.20f,
                                     float min_solidity = 0.50f,
                                     float min_green_ratio = 0.50f)
  {
    if (label.empty() || bgrImg.empty())
      return;
    if (label.size() != bgrImg.size())
      return;

    const int img_h = label.rows;
    const int img_w = label.cols;

    // ROI 范围: 50% ~ 90% 高度
    const int roi_y_top = static_cast<int>(img_h * 0.5);
    const int roi_y_bot = static_cast<int>(img_h * 0.9);

    // Step 0: BGR → HSV
    cv::Mat blurred;
    cv::GaussianBlur(bgrImg, blurred, cv::Size(5, 5), 0);
    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // Step 0.5: 绿色前置条件 — 严格绿色 H∈[25,85]
    //   黄草 H≈15-25 不算绿色, 只有真正的绿草才算
    cv::Mat strict_green;
    cv::inRange(hsv, cv::Scalar(25, 40, 40), cv::Scalar(85, 255, 255),
                strict_green);
    cv::Mat roi_green = strict_green(cv::Range(roi_y_top, roi_y_bot),
                                      cv::Range(0, img_w));
    int green_pixels = cv::countNonZero(roi_green);
    int total_roi_pixels = roi_green.total();
    float green_ratio = (total_roi_pixels > 0)
                             ? static_cast<float>(green_pixels) / total_roi_pixels
                             : 0.0f;

    if (green_ratio < min_green_ratio)
    {
      // 绿色不足 (如枯黄草地), 跳过检测
      return;
    }

    // Step 1: 非植被色 = NOT 植被色 H∈[15,85]
    cv::Mat veg_mask;
    cv::inRange(hsv, cv::Scalar(15, 30, 40), cv::Scalar(85, 255, 255), veg_mask);
    cv::Mat non_veg;
    cv::bitwise_not(veg_mask, non_veg);

    // 饱和度/亮度约束: (S > 25 OR V > 150) AND V > 40
    cv::Mat hsv_channels[3];
    cv::split(hsv, hsv_channels);
    cv::Mat sat_ok = hsv_channels[1] > 25;
    cv::Mat bright_ok = hsv_channels[2] > 150;
    cv::Mat sat_val_ok = sat_ok | bright_ok;
    cv::Mat not_too_dark = hsv_channels[2] > 40;
    sat_val_ok &= not_too_dark;
    non_veg &= sat_val_ok;

    // Step 2: 语义约束 — 只处理草坪(2)或道路(3)
    cv::Mat mask_label2, mask_label3, lawn_road_mask;
    cv::compare(label, 2, mask_label2, cv::CMP_EQ);
    cv::compare(label, 3, mask_label3, cv::CMP_EQ);
    lawn_road_mask = mask_label2 | mask_label3;
    cv::Mat candidate = non_veg & lawn_road_mask;

    // Step 2.5: ROI 垂直范围约束
    candidate(cv::Range(0, roi_y_top), cv::Range(0, img_w)).setTo(0);
    candidate(cv::Range(roi_y_bot, img_h), cv::Range(0, img_w)).setTo(0);

    // Step 3: 形态学去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(candidate, candidate, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, kernel);

    // Step 4: 连通域分析
    cv::Mat cc_labels, cc_stats, cc_centroids;
    int n_labels = cv::connectedComponentsWithStats(candidate, cc_labels, cc_stats,
                                                     cc_centroids, 8, CV_32S);

    cv::Mat gray;
    cv::cvtColor(bgrImg, gray, cv::COLOR_BGR2GRAY);

    cv::Mat result_mask = cv::Mat::zeros(label.size(), CV_8UC1);

    for (int i = 1; i < n_labels; i++)
    {
      int area = cc_stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < min_area)
        continue;

      int x = cc_stats.at<int>(i, cv::CC_STAT_LEFT);
      int y = cc_stats.at<int>(i, cv::CC_STAT_TOP);
      int w = cc_stats.at<int>(i, cv::CC_STAT_WIDTH);
      int h = cc_stats.at<int>(i, cv::CC_STAT_HEIGHT);

      cv::Rect roi_rect(x, y, w, h);
      cv::Mat comp_mask_roi = (cc_labels(roi_rect) == i);
      cv::Mat gray_roi = gray(roi_rect);

      // 特征 A: 直线边缘评分
      float edge_score = computeStraightEdgeScore(gray_roi, comp_mask_roi);

      // 特征 B: 凸度
      float solidity = 0.0f;
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(comp_mask_roi.clone(), contours,
                       cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      if (!contours.empty())
      {
        auto &contour = *std::max_element(
            contours.begin(), contours.end(),
            [](const auto &a, const auto &b)
            { return cv::contourArea(a) < cv::contourArea(b); });
        double contour_area = cv::contourArea(contour);
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        if (hull_area > 0)
        {
          solidity = static_cast<float>(contour_area / hull_area);
        }
      }

      // 决策: 大面积区域适当放宽边缘阈值
      float eff_edge_thresh = min_edge_score;
      if (area > min_area * 5)
      {
        eff_edge_thresh *= 0.6f;
      }

      if (edge_score >= eff_edge_thresh && solidity >= min_solidity)
      {
        result_mask.setTo(255, cc_labels == i);
      }
    }

    // Step 5: 应用结果 - 将识别出的障碍物标记为label=5(静态障碍物)
    label.setTo(5, result_mask);
  }

  // 基于检测框的深度补全函数
  Mat depthInpaintingByDetections(const Mat &depth, const Mat &label_map,
                                   const std::vector<Detection> &detections)
  {
    Mat depth_inpainted = depth.clone();

    //     cout << "\n========== 基于检测框的深度补全 ==========" << endl;

    // 对每个检测框进行处理
    for (size_t i = 0; i < detections.size(); i++) {
      const Detection &det = detections[i];

      // 只处理障碍物类别（id=4对应label=104）
      if (det.id != 4) continue;

      // 获取检测框区域
      int xmin = std::max(0, static_cast<int>(det.bbox.xmin));
      int ymin = std::max(0, static_cast<int>(det.bbox.ymin));
      int xmax = std::min(depth.cols, static_cast<int>(det.bbox.xmax));
      int ymax = std::min(depth.rows, static_cast<int>(det.bbox.ymax));

      if (xmax <= xmin || ymax <= ymin) continue;

      cv::Rect det_rect(xmin, ymin, xmax - xmin, ymax - ymin);

    //       cout << "\n[检测框 " << i << "] 位置: " << det_rect << endl;

      // 统计检测框内的深度情况
      int total_pixels = det_rect.width * det_rect.height;
      int valid_depth_pixels = 0;
      int invalid_depth_pixels = 0;
      std::vector<float> valid_depths;

      for (int y = ymin; y < ymax; y++) {
        for (int x = xmin; x < xmax; x++) {
          float d = depth.at<float>(y, x);
          if (d > 0 && d < 9.5f) {
            valid_depth_pixels++;
            valid_depths.push_back(d);
          } else {
            invalid_depth_pixels++;
          }
        }
      }

      // 如果无效深度比例超过20%，进行补全
      if (invalid_depth_pixels > total_pixels * 0.2) {
        // 方法1：使用检测框边界的深度中值
        std::vector<float> boundary_depths;
        int boundary_width = 10;

        // 收集边界区域的有效深度
        for (int y = std::max(0, ymin - boundary_width);
             y < std::min(depth.rows, ymax + boundary_width); y++) {
          for (int x = std::max(0, xmin - boundary_width);
               x < std::min(depth.cols, xmax + boundary_width); x++) {

            // 只收集边界附近的点
            bool is_boundary = (y < ymin || y >= ymax || x < xmin || x >= xmax);

            if (is_boundary) {
              float d = depth.at<float>(y, x);
              if (d > 0 && d < 9.5f) {
                boundary_depths.push_back(d);
              }
            }
          }
        }

        // 计算填充深度值
        float fill_depth = 0;
        if (!boundary_depths.empty()) {
          std::sort(boundary_depths.begin(), boundary_depths.end());
          fill_depth = boundary_depths[boundary_depths.size() / 2]; // 中值
        } else if (!valid_depths.empty()) {
          // 如果边界没有有效深度，使用检测框内部的中值
          std::sort(valid_depths.begin(), valid_depths.end());
          fill_depth = valid_depths[valid_depths.size() / 2];
        }

        if (fill_depth > 0) {
          // 填充检测框内的无效深度
          int filled = 0;
          for (int y = ymin; y < ymax; y++) {
            for (int x = xmin; x < xmax; x++) {
              // 只填充label=104的像素
              if (label_map.at<uchar>(y, x) == 104) {
                float d = depth_inpainted.at<float>(y, x);
                if (d <= 0 || d >= 9.5f) {
                  depth_inpainted.at<float>(y, x) = fill_depth;
                  filled++;
                }
              }
            }
          }
        }
      }
    }

    return depth_inpainted;
  }

  // 深度补全函数：针对障碍物区域（Label >= 100）填充稀疏深度
  // 增强版：基于分割图连通区域的深度补全
  Mat depthInpaintingForObstacles(const Mat &depth, const Mat &label_map)
  {
    Mat depth_inpainted = depth.clone();

    // 1. 创建障碍物掩码（Label >= 100）
    Mat obstacle_mask = Mat::zeros(label_map.size(), CV_8UC1);
    for (int y = 0; y < label_map.rows; y++) {
      for (int x = 0; x < label_map.cols; x++) {
        uint8_t label = label_map.at<uchar>(y, x);
        // 识别障碍物：label >= 100 或 label == 4 或 label == 5
        if (label >= 100 || label == 4 || label == 5) {
          obstacle_mask.at<uchar>(y, x) = 255;
        }
      }
    }

    int obstacle_pixels = cv::countNonZero(obstacle_mask);

    if (obstacle_pixels == 0) {
      return depth_inpainted;
    }

    // 2. 查找障碍物的连通区域
    Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(obstacle_mask, labels, stats, centroids);

    // 3. 对每个连通区域进行深度补全
    int total_regions_processed = 0;
    int total_pixels_filled = 0;

    for (int comp = 1; comp < num_components; comp++) {
      // 获取连通区域的统计信息
      int left = stats.at<int>(comp, cv::CC_STAT_LEFT);
      int top = stats.at<int>(comp, cv::CC_STAT_TOP);
      int width = stats.at<int>(comp, cv::CC_STAT_WIDTH);
      int height = stats.at<int>(comp, cv::CC_STAT_HEIGHT);
      int area = stats.at<int>(comp, cv::CC_STAT_AREA);

      // 跳过太小的区域（可能是噪声）
      if (area < 50) continue;

      cv::Rect roi(left, top, width, height);

      // 统计该区域的深度情况
      int valid_depth_count = 0;
      int invalid_depth_count = 0;
      std::vector<float> valid_depths;
      valid_depths.reserve(area);

      for (int y = top; y < top + height && y < depth.rows; y++) {
        for (int x = left; x < left + width && x < depth.cols; x++) {
          if (labels.at<int>(y, x) == comp) {
            float d = depth.at<float>(y, x);
            if (d > 0 && d < 9.5f) {
              valid_depth_count++;
              valid_depths.push_back(d);
            } else {
              invalid_depth_count++;
            }
          }
        }
      }

      int total_pixels = valid_depth_count + invalid_depth_count;
      float invalid_ratio = (float)invalid_depth_count / total_pixels;

      // 如果无效深度比例超过20%，进行补全
      if (invalid_ratio > 0.2) {
        // 收集边界深度信息
        std::vector<float> boundary_depths;
        boundary_depths.reserve(200);
        int boundary_width = 10;

        // 扩展边界区域收集深度
        for (int y = std::max(0, top - boundary_width);
             y < std::min(depth.rows, top + height + boundary_width); y++) {
          for (int x = std::max(0, left - boundary_width);
               x < std::min(depth.cols, left + width + boundary_width); x++) {

            // 只收集边界附近的点（不在当前连通区域内）
            bool is_boundary = (labels.at<int>(y, x) != comp);

            if (is_boundary) {
              float d = depth.at<float>(y, x);
              if (d > 0 && d < 9.5f) {
                boundary_depths.push_back(d);
              }
            }
          }
        }

        // 如果边界深度不足，扩大搜索范围
        if (boundary_depths.size() < 20) {
          boundary_width = 20;
          for (int y = std::max(0, top - boundary_width);
               y < std::min(depth.rows, top + height + boundary_width); y++) {
            for (int x = std::max(0, left - boundary_width);
                 x < std::min(depth.cols, left + width + boundary_width); x++) {
              if (labels.at<int>(y, x) != comp) {
                float d = depth.at<float>(y, x);
                if (d > 0 && d < 9.5f) {
                  boundary_depths.push_back(d);
                }
              }
            }
          }
        }

        // 计算填充深度值
        float fill_depth = 0;
        if (!boundary_depths.empty()) {
          std::sort(boundary_depths.begin(), boundary_depths.end());
          fill_depth = boundary_depths[boundary_depths.size() / 2]; // 中值
        } else if (!valid_depths.empty()) {
          std::sort(valid_depths.begin(), valid_depths.end());
          fill_depth = valid_depths[valid_depths.size() / 2];
        }

        // 填充该区域内的无效深度
        int region_filled = 0;
        if (fill_depth > 0) {
          for (int y = top; y < top + height && y < depth.rows; y++) {
            for (int x = left; x < left + width && x < depth.cols; x++) {
              if (labels.at<int>(y, x) == comp) {
                float d = depth_inpainted.at<float>(y, x);
                if (d <= 0 || d >= 9.5f) {
                  depth_inpainted.at<float>(y, x) = fill_depth;
                  region_filled++;
                }
              }
            }
          }
          total_pixels_filled += region_filled;
          total_regions_processed++;
        }
      }
    }

    // 4. 多轮迭代填充剩余的无效像素
    Mat invalid_depth_mask = Mat::zeros(label_map.size(), CV_8UC1);
    int invalid_count = 0;
    for (int y = 0; y < depth.rows; y++) {
      for (int x = 0; x < depth.cols; x++) {
        if (obstacle_mask.at<uchar>(y, x) == 255) {
          float d = depth_inpainted.at<float>(y, x);
          if (d <= 0 || d >= 9.5f) {
            invalid_depth_mask.at<uchar>(y, x) = 255;
            invalid_count++;
          }
        }
      }
    }

    if (invalid_count > 0) {
      int filled_count = 0;
      int max_iterations = 3;
      int kernel_size = 5;

      for (int iter = 0; iter < max_iterations; iter++) {
        int iter_filled = 0;
        Mat temp_depth = depth_inpainted.clone();

        for (int y = 0; y < depth.rows; y++) {
          for (int x = 0; x < depth.cols; x++) {
            if (invalid_depth_mask.at<uchar>(y, x) == 255) {
              std::vector<float> valid_depths;
              std::vector<float> weights;
              valid_depths.reserve(121);
              weights.reserve(121);

              for (int dy = -kernel_size; dy <= kernel_size; dy++) {
                for (int dx = -kernel_size; dx <= kernel_size; dx++) {
                  int ny = y + dy;
                  int nx = x + dx;

                  if (ny >= 0 && ny < depth.rows && nx >= 0 && nx < depth.cols) {
                    if (obstacle_mask.at<uchar>(ny, nx) == 255) {
                      float d = depth_inpainted.at<float>(ny, nx);
                      if (d > 0 && d < 9.5f) {
                        valid_depths.push_back(d);
                        float dist = sqrt(dx*dx + dy*dy);
                        weights.push_back(1.0f / (dist + 1.0f));
                      }
                    }
                  }
                }
              }

              if (valid_depths.size() >= 3) {
                float weighted_sum = 0;
                float weight_sum = 0;
                for (size_t i = 0; i < valid_depths.size(); i++) {
                  weighted_sum += valid_depths[i] * weights[i];
                  weight_sum += weights[i];
                }
                float filled_depth = weighted_sum / weight_sum;

                temp_depth.at<float>(y, x) = filled_depth;
                invalid_depth_mask.at<uchar>(y, x) = 0;
                iter_filled++;
                filled_count++;
              }
            }
          }
        }

        depth_inpainted = temp_depth;
        if (iter_filled == 0) break;
      }
    }

    return depth_inpainted;
  }

  ProcessResult processMode0(const Mat &left, const Mat &right,
                             const Mat &grayImageL, const Mat &grayImageR)
  {
    (void)left; (void)right; (void)grayImageL; (void)grayImageR;
    ProcessResult result;
    result.mode_name = "Disable";
    cout << "[Mode 0] Disable - no processing" << endl;
    return result;
  }

  // Mode 1: Only Depth - 仅深度，无DL模型
  ProcessResult processMode1(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR)
  {
    (void)right;
    ProcessResult result;
    result.mode_name = "OnlyDepth";

    cout << "[Mode 1] Only Depth - no DL model" << endl;

    auto depth_start = chrono::high_resolution_clock::now();

    // 计算深度
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud;
    Mat disparity =
        stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
    result.depth_map = stereo_multi_match.stereo_multi_process_filter(
        disparity, Mat(), config_.enable_height_filter_);

    // 生成点云 (仅深度，无语义标签)
    Mat left_copy = left.clone();
    stereo_multi_match.stereo_process_pc_rgbl_depth(
        result.depth_map, left_copy, xyz_rgbl_cloud, result.point_cloud);

    auto depth_end = chrono::high_resolution_clock::now();
    double depth_time =
        chrono::duration<double, milli>(depth_end - depth_start).count();

    cout << "[Step 1/1] Depth computation done: " << depth_time << " ms"
         << endl;

    return result;
  }

  // Mode 2: Detection fusion depth - 检测融合深度
  ProcessResult processMode2(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR)
  {
    (void)right;
    ProcessResult result;
    result.mode_name = "Detection";

    cout << "[Mode 2] Detection fusion depth" << endl;

    // Step 1: 检测
    auto det_start = chrono::high_resolution_clock::now();
    std::vector<Detection> detections;
    Mat left_copy = left.clone();
    detPerception_.perception_process(left_copy);
    detPerception_.perception_postprocess_nanodet(left_copy, detections);
    detPerception_.task_release();
    auto det_end = chrono::high_resolution_clock::now();
    cout << "[Step 1/3] Detection done: " << detections.size() << " objects, "
         << chrono::duration<double, milli>(det_end - det_start).count()
         << " ms" << endl;

    // Step 2: 深度计算
    auto depth_start = chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud;
    Mat disparity =
        stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
    result.depth_map = stereo_multi_match.stereo_multi_process_filter(
        disparity, Mat(), config_.enable_height_filter_);
    auto depth_end = chrono::high_resolution_clock::now();
    cout << "[Step 2/3] Depth computation done: "
         << chrono::duration<double, milli>(depth_end - depth_start).count()
         << " ms" << endl;

    // Step 3: 融合（可选的点云输出）
    auto fusion_start = chrono::high_resolution_clock::now();
    Mat left_copy2 = left.clone();
    stereo_multi_match.stereo_process_pc_rgbl_dest(result.depth_map, left_copy2,
                                                   detections, xyz_rgbl_cloud,
                                                   result.point_cloud);
    auto fusion_end = chrono::high_resolution_clock::now();
    cout << "[Step 3/3] Fusion done: "
         << chrono::duration<double, milli>(fusion_end - fusion_start).count()
         << " ms" << endl;

    return result;
  }

  // Mode 3: Segmentation fusion depth - 语义分割融合深度
  ProcessResult processMode3(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR)
  {
    (void)right;
    ProcessResult result;
    result.mode_name = "Segmentation";

    cout << "[Mode 3] Segmentation fusion depth" << endl;

    // Step 1: 语义分割
    auto seg_start = chrono::high_resolution_clock::now();
    Mat left_copy = left.clone();
    result.label_map = segPerception_.perception_postprocess_int64_erode(
        left_copy, config_.erode_pixel);
    auto seg_end = chrono::high_resolution_clock::now();
    cout << "[Step 1/3] Segmentation done: "
         << chrono::duration<double, milli>(seg_end - seg_start).count()
         << " ms" << endl;

    // Step 2: 深度计算
    auto depth_start = chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud;
    Mat disparity =
        stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
    result.depth_map = stereo_multi_match.stereo_multi_process_filter(
        disparity, result.label_map, config_.enable_height_filter_);
    auto depth_end = chrono::high_resolution_clock::now();
    cout << "[Step 2/3] Depth computation done: "
         << chrono::duration<double, milli>(depth_end - depth_start).count()
         << " ms" << endl;

    // Step 3: 融合
    auto fusion_start = chrono::high_resolution_clock::now();
    Mat left_copy2 = left.clone();
    stereo_multi_match.stereo_process_pci_depth_rgb_seg_fusion(
        result.depth_map, result.label_map, left_copy2, xyz_rgbl_cloud,
        result.point_cloud);
    auto fusion_end = chrono::high_resolution_clock::now();
    cout << "[Step 3/3] Fusion done: "
         << chrono::duration<double, milli>(fusion_end - fusion_start).count()
         << " ms" << endl;

    return result;
  }

  // Mode 4: Charge station QR recognition - 充电桩二维码识别
  ProcessResult processMode4(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR)
  {
    (void)left; (void)right;
    ProcessResult result;
    result.mode_name = "ChargeStationQR";

    cout << "[Mode 4] Charge station QR recognition" << endl;
    cout << "[Note] Mode 4 requires ArUco detection - simplified for offline "
            "version"
         << endl;

    // 简化版本：仅计算深度
    Mat disparity =
        stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
    result.depth_map = stereo_multi_match.stereo_multi_process_filter(
        disparity, Mat(), config_.enable_height_filter_);

    return result;
  }

  // Mode 5: Multi-task recognition - 多任务识别 (分割+检测)
  ProcessResult processMode5(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR)
  {
    (void)right;
    ProcessResult result;
    result.mode_name = "MultiTask";

    cout << "[Mode 5] Multi-task recognition (seg + det)" << endl;

    // Step 1: 多任务推理
    auto infer_start = chrono::high_resolution_clock::now();
    // cv::Mat croppedImg = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
    // std::vector<Detection> detections;
    // Mat left_copy = left.clone();

    cv::Rect cropRegion(0, 0, left.cols, 432);
    cv::Mat croppedImg = left(cropRegion);
    // 灰度图不再裁剪，视差图和深度图在 640x480 原始尺寸下计算

    Mat resizeImg = cv::Mat::zeros(384, 640, croppedImg.type());
    cv::resize(croppedImg, resizeImg, cv::Size(640, 384));

    cv::Mat dst_label(resizeImg.rows, resizeImg.cols, CV_8UC1);
    std::vector<Detection> detections, dect_dst, ct_dect_src;
    cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
    Mat lab_out, lab_temp;

    Rect cdt_rect;
    if (config_.enabel_cdt)
    {
      ct_dect_src.clear();
      // cdt 模型在 432 下推演，返回的是 640x432 尺度下的检测框
      cdtPerception_.perception_process_bgr(croppedImg, ct_dect_src);
      cdt_rect = get_cdt_rect(ct_dect_src);
      cout << "cdt_rect: " << cdt_rect << endl;
      cout << "cdt_rect.area: " << cdt_rect.area() << endl;
    }

    // 1. 网络在 384 下推理
    multiPerception_.perception_process_bgr_no_argmax_erode_mul(
        resizeImg, detections, img_label, config_.erode_pixel);
    filterLabelDect(img_label, detections, dst_label, dect_dst);

    // 红色砖头颜色后处理: 在640x384尺度上修正被误分类为草坪/道路的红色区域
    if (config_.enable_red_brick_refine_)
    {
      refineObstacleByColorAndEdge(dst_label, resizeImg, config_.red_brick_min_area_);
    }

    // 2. 将标签图还原回 432，使用最近邻插值，防止产生浮点类别
    cv::Mat label_432;
    cv::resize(dst_label, label_432, cv::Size(640, 432), 0, 0,
               cv::INTER_NEAREST);

    // 3. 将 384 尺度下获取的检测框还原回 432
    for (auto &det : dect_dst)
    {
      det.bbox.ymin =
          std::max(0, static_cast<int>(det.bbox.ymin * (432.0f / 384.0f)));
      det.bbox.ymax =
          std::min(432, static_cast<int>(det.bbox.ymax * (432.0f / 384.0f)));
    }

    // 4. 将 cdt 遮罩施加在放大后的 label_432 上，实现坐标系严丝合缝
    if (config_.enabel_cdt && cdt_rect.area() > 0)
    {
      label_432(cdt_rect) = -1;
    }

    // printLabelDistribution(img_label);  // 直接打印分布

    auto infer_end = chrono::high_resolution_clock::now();
    cout << "[Step 1/4] Multi-task inference done: "
         << chrono::duration<double, milli>(infer_end - infer_start).count()
         << " ms" << endl;
    // cout << "  - Detections: " << detections.size() << " objects" << endl;

    // Step 2: 标签后处理 (简化版，跳过复杂的label processing)
    result.label_map = label_432.clone(); // 此后点云融合只使用 640x432 的 label
    cout << "[Step 2/4] Label processing done (simplified)" << endl;

    // Step 3: 深度计算 (视差图和深度图在 640x480 原始尺寸下计算)
    auto depth_start = chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud, out_xyz_rgbl_cloud;
    Mat disparity =
        stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
    // 构造 640x480 的标签图，底部 48 行用背景值(2)填充，用于深度滤波
    cv::Mat label_480 = cv::Mat(480, 640, CV_8UC1, cv::Scalar(2));
    label_432.copyTo(label_480(cv::Rect(0, 0, 640, 432)));
    Mat depth_480 = stereo_multi_match.stereo_multi_process_filter(
        disparity, label_480, config_.enable_height_filter_);
    // 裁剪深度图到 640x432 (y轴 0~432)，用于后续融合
    cv::Mat depth_432 = depth_480(cv::Rect(0, 0, 640, 432)).clone();

    auto depth_end = chrono::high_resolution_clock::now();
    cout << "[Step 3/4] Depth computation done: "
         << chrono::duration<double, milli>(depth_end - depth_start).count()
         << " ms" << endl;

    // Step 3.5: 深度补全
    auto inpaint_start = chrono::high_resolution_clock::now();
    Mat depth_inpainted_432 = depth_432.clone();
    if (config_.depth_inpainting_strategy_ > 0 && !depth_432.empty())
    {
      if (config_.depth_inpainting_strategy_ == 1 || config_.depth_inpainting_strategy_ == 3)
      {
        depth_inpainted_432 = depthInpaintingByDetections(depth_inpainted_432, result.label_map, dect_dst);
      }
      if (config_.depth_inpainting_strategy_ == 2 || config_.depth_inpainting_strategy_ == 3)
      {
        depth_inpainted_432 = depthInpaintingForObstacles(depth_inpainted_432, result.label_map);
      }
    }
    auto inpaint_end = chrono::high_resolution_clock::now();
    cout << "[Step 3.5/4] Depth inpainting done: "
         << chrono::duration<double, milli>(inpaint_end - inpaint_start).count()
         << " ms" << endl;

    // Step 4: 融合 (在 640x432 尺度下，使用裁剪后的深度图)
    auto fusion_start = chrono::high_resolution_clock::now();
    stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
        depth_inpainted_432, result.label_map, dect_dst, croppedImg, xyz_rgbl_cloud,
        out_xyz_rgbl_cloud);

    if (config_.m_enable_debug_show)
    {
      Mat img_seg_show;
      // 绘制分割结果，dect_dst 已是 432 高度比例，绘制在 croppedImg(640x432) 上
      Mat pure_seg_mat =
          drawResultOptimized(croppedImg, label_432, dect_dst, img_seg_show);

      Mat origin_seg;
      cv::hconcat(croppedImg, pure_seg_mat, origin_seg); // 拼接图片1和图片2
      cv::hconcat(origin_seg, img_seg_show, origin_seg); // 拼接图片1和图片2
      Mat xyz_rgbl, final_compared;
      stereo_point_cloud stereoPointCloud;
      stereoPointCloud.show_xyz_rgbl_plane_point_cloud_final(out_xyz_rgbl_cloud,
                                                             xyz_rgbl);
      cv::vconcat(origin_seg, xyz_rgbl,
                  final_compared); // 1920x432 + 1920x960 = 1920x1392

      // 构造输出路径: finalPicDir + 原文件名 + "_cdt.jpg"
      string finalPicPath =
          config_.finalPicDir + current_image_name_ + "_mul_cdt.jpg";
      imwrite(finalPicPath, final_compared);

      string finalPcdPath =
          config_.finalPcdDir + current_image_name_ + "_mul_cdt";
      savePcdfile_with_rgb_label(out_xyz_rgbl_cloud, finalPcdPath);
    }
    auto fusion_end = chrono::high_resolution_clock::now();
    cout << "[Step 4/4] Fusion done: "
         << chrono::duration<double, milli>(fusion_end - fusion_start).count()
         << " ms" << endl;

    return result;
  }

  // Mode 7: DSG Night recognition - DSG 夜间识别
  ProcessResult processMode7(const Mat &left, const Mat &right, Mat grayImageL, Mat grayImageR)
  {
    (void)right;
    ProcessResult result;
    result.mode_name = "DSG_Night";

    cout << "[Mode 7] DSG Night recognition" << endl;

    cv::Rect cropRegion(0, 0, left.cols, 432);
    cv::Mat croppedImg = left(cropRegion);

    // DSG logic: Align with reference run_cdt_dsg_fusion_dir.cpp

    std::vector<Detection> dect_src, dect_dst, cdt_src;
    cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 1;
    cv::Mat lab_out, lab_dst;

    // 1. Resize to 640x384 for segmentation inference
    cv::Mat resizedForSeg;
    cv::resize(croppedImg, resizedForSeg, cv::Size(640, 384));

    // Set inference internal dimensions
    dsgPerception_.ori_height = resizedForSeg.rows;
    dsgPerception_.ori_width  = resizedForSeg.cols;

    // 2. Inference Multi-task (Detection + Segmentation)
    dsgPerception_.perception_process_bgr_no_argmax_erode(
        resizedForSeg, dect_src, img_label, lab_out, config_.erode_pixel);

    // 2. Resize segmentation result from 640x384 back to 640x432
    cv::resize(lab_out, lab_out, cv::Size(640, 432), 0, 0, cv::INTER_NEAREST);

    // 3. Map detection IDs and prepare lab_dst (logic from reference line 1130-1137)
    lab_out.copyTo(lab_dst);
    dect_dst.clear();
    for (const auto& det : dect_src) {
        Detection fixed_det = det;
        fixed_det.id += 100; // Map to 100+ format
        dect_dst.push_back(fixed_det);
    }

    // CES 逻辑已关闭
    // bool all_neighbors_are_lawn = false;
    // this->processLabelContours(lab_dst, all_neighbors_are_lawn);

    // 偏黑区域标记为可行走区域（类别3：road），通过 bool 变量控制是否启用
    bool enable_hsv_dark_filter = false;
    if (enable_hsv_dark_filter)
    {
        cv::Mat hsvImg;
        cv::cvtColor(croppedImg, hsvImg, cv::COLOR_BGR2HSV);

        cv::Scalar lowerBlack(0, 0, 0);      // H, S, V
        cv::Scalar upperBlack(180, 50, 80);  // 低饱和度、低亮度

        cv::Mat darkMask;
        cv::inRange(hsvImg, lowerBlack, upperBlack, darkMask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(darkMask, darkMask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(darkMask, darkMask, cv::MORPH_OPEN, kernel);

        cv::Mat resizedMask;
        cv::resize(darkMask, resizedMask, lab_dst.size(), 0, 0, cv::INTER_NEAREST);
        lab_dst.setTo(3, resizedMask);
    }

    // Handle CDT if enabled
    if (config_.enabel_cdt)
    {
      cdtPerception_.perception_process_bgr(croppedImg, cdt_src);
      cv::Rect cdt_rect = get_cdt_rect(cdt_src);
      if (cdt_rect.area() > 0)
      {
        lab_dst(cdt_rect) = -1;
      }
    }

    result.label_map = lab_dst.clone();

    if (grayImageR.empty())
      return result;

    // Depth computation
    auto depth_start = chrono::high_resolution_clock::now();
    Mat depth_cal = stereo_multi_match.stereo_multi_process(grayImageL, grayImageR, false);

    // Crop depth to 432 height to match croppedImg/labels
    cv::Mat depth_432 = depth_cal(cv::Rect(0, 0, 640, 432)).clone();

    auto depth_end = chrono::high_resolution_clock::now();
    cout << "[Step 2/3] Depth computation done: "
         << chrono::duration<double, milli>(depth_end - depth_start).count() << " ms" << endl;

    // Fusion
    auto fusion_start = chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbi_cloud, out_xyz_rgbi_cloud;
    stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
        depth_432, lab_dst, dect_dst, croppedImg, xyz_rgbi_cloud, out_xyz_rgbi_cloud);
    result.point_cloud = out_xyz_rgbi_cloud;
    auto fusion_end = chrono::high_resolution_clock::now();
    cout << "[Step 3/3] Fusion done: "
         << chrono::duration<double, milli>(fusion_end - fusion_start).count() << " ms" << endl;

    // string labelPcdPath = savePcdDirPath + name + "_rgbl";
    // savePcdfile_with_rgb_label(out_xyz_rgbi_cloud, labelPcdPath);

    if (config_.m_enable_debug_show)
    {
      Mat img_seg_show, xyz_rgbl_show, origin_seg, pure_seg;

      // 可视化统一使用 640x432（与 croppedImg 原始尺寸一致）
      cv::Mat croppedImgVis;
      cv::resize(croppedImg, croppedImgVis, cv::Size(640, 432));
      cv::Mat labVis;
      cv::resize(lab_dst, labVis, cv::Size(640, 432), 0, 0, cv::INTER_NEAREST);

      bool enable_draw_box = false; // 控制是否显示检测框，默认不显示
      pure_seg = drawResultOptimized(croppedImgVis, labVis, dect_dst, img_seg_show, enable_draw_box);

      // 上方：原图 | 纯色分割 | 叠加图 → 1920x432
      cv::hconcat(croppedImgVis, pure_seg, origin_seg);
      cv::hconcat(origin_seg, img_seg_show, origin_seg);

      // 下方：三视图（label行 480 + RGB行 480 → 1920x960）
      stereo_point_cloud spc;
      spc.show_xyz_rgbl_plane_point_cloud_final(out_xyz_rgbi_cloud, xyz_rgbl_show);

      // 最终拼接：432 + 480 + 480 = 1392 高度
      cv::Mat final_compared;
      cv::vconcat(origin_seg, xyz_rgbl_show, final_compared);

      string finalPicPath = config_.finalPicDir + current_image_name_ + "_dsg_night.jpg";
      imwrite(finalPicPath, final_compared);

      string finalPcdPath = config_.finalPcdDir + current_image_name_ + "_dsg_night";
      savePcdfile_with_rgb_label(out_xyz_rgbi_cloud, finalPcdPath);
    }

    return result;
  }

  // Mode 6: Sub-task recognition - 子任务识别
  ProcessResult processMode6(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR)
  {
    (void)right;
    ProcessResult result;
    result.mode_name = "SubTask";

    cout << "[Mode 6] Sub-task recognition" << endl;

    // Step 1: 子任务推理
    auto infer_start = chrono::high_resolution_clock::now();

    cv::Mat croppedImg, resizeImg;
    if (left.rows == 480)
    {
      cv::Rect cropRegion(0, 0, left.cols, 432);
      croppedImg = left(cropRegion);
      resizeImg = cv::Mat::zeros(384, 640, croppedImg.type());
      cv::resize(croppedImg, resizeImg, cv::Size(640, 384));
    }
    else if (left.rows == 384)
    {
      croppedImg = left;
      resizeImg = left.clone();
    }

    cv::Mat dst_label(resizeImg.rows, resizeImg.cols, CV_8UC1);
    std::vector<Detection> detections, dect_dst;
    cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
    Mat lab_out = cv::Mat::zeros(384, 640, CV_8UC1);

    mulSubPerception.perception_process_bgr_no_argmax_erode(
        resizeImg, detections, img_label, lab_out, config_.erode_pixel);

    filterLabelDect(lab_out, detections, dst_label, dect_dst);

    // 红色砖头颜色后处理: 在640x384尺度上修正被误分类为草坪/道路的红色区域
    if (config_.enable_red_brick_refine_)
    {
      refineObstacleByColorAndEdge(dst_label, resizeImg, config_.red_brick_min_area_);
    }

    // 还原 label 至真实尺寸
    cv::Mat label_432;
    if (dst_label.rows == 384 && croppedImg.rows == 432)
    {
      cv::resize(dst_label, label_432, cv::Size(640, 432), 0, 0,
                 cv::INTER_NEAREST);
      for (auto &det : dect_dst)
      {
        det.bbox.ymin =
            std::max(0, static_cast<int>(det.bbox.ymin * (432.0f / 384.0f)));
        det.bbox.ymax =
            std::min(432, static_cast<int>(det.bbox.ymax * (432.0f / 384.0f)));
      }
    }
    else
    {
      label_432 = dst_label.clone();
    }

    auto infer_end = chrono::high_resolution_clock::now();
    cout << "[Step 1/3] Sub-task inference done: "
         << chrono::duration<double, milli>(infer_end - infer_start).count()
         << " ms" << endl;

    result.label_map = label_432.clone();

    // Step 2: 深度计算与融合
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud, out_xyz_rgbl_cloud;

    if (!grayImageR.empty())
    {
      auto depth_start = chrono::high_resolution_clock::now();

      Mat disparity =
          stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);

      // label_map 从 640x432 pad 到 640x480，底部填充 2（背景标签）
      cv::Mat label_480 = cv::Mat::zeros(480, 640, result.label_map.type()) + 2;
      result.label_map.copyTo(label_480(cv::Rect(0, 0, 640, 432)));

      Mat depth_480 = stereo_multi_match.stereo_multi_process_filter(
          disparity, label_480, config_.enable_height_filter_);

      // 裁剪 depth 到 640x432 用于融合
      cv::Rect depthCropRegion(0, 0, 640, 432);
      Mat depth_cal = depth_480(depthCropRegion);

      auto depth_end = chrono::high_resolution_clock::now();
      cout << "[Step 2/4] Depth computation done: "
           << chrono::duration<double, milli>(depth_end - depth_start).count()
           << " ms" << endl;

      // Step 3: 深度补全
      auto inpaint_start = chrono::high_resolution_clock::now();
      Mat depth_inpainted = depth_cal.clone();

      // 深度补全策略
      if (config_.depth_inpainting_strategy_ > 0 && !depth_cal.empty())
      {
        // 策略1或3: 基于检测框的补全
        if (config_.depth_inpainting_strategy_ == 1 || config_.depth_inpainting_strategy_ == 3)
        {
          depth_inpainted = depthInpaintingByDetections(depth_inpainted, result.label_map, dect_dst);
        }

        // 策略2或3: 基于语义分割的补全
        if (config_.depth_inpainting_strategy_ == 2 || config_.depth_inpainting_strategy_ == 3)
        {
          depth_inpainted = depthInpaintingForObstacles(depth_inpainted, result.label_map);
        }
      }

      auto inpaint_end = chrono::high_resolution_clock::now();
      cout << "[Step 3/4] Depth inpainting done: "
           << chrono::duration<double, milli>(inpaint_end - inpaint_start).count()
           << " ms" << endl;

      // Step 4: 融合
      auto fusion_start = chrono::high_resolution_clock::now();
      stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
          depth_inpainted, result.label_map, dect_dst, croppedImg, xyz_rgbl_cloud,
          out_xyz_rgbl_cloud);

      auto fusion_end = chrono::high_resolution_clock::now();
      cout << "[Step 4/4] Fusion done: "
           << chrono::duration<double, milli>(fusion_end - fusion_start).count()
           << " ms" << endl;
    }

    if (config_.m_enable_debug_show)
    {
      Mat img_seg_show;
      Mat pure_seg_mat = drawResultOptimized(croppedImg, result.label_map,
                                             dect_dst, img_seg_show);
      Mat origin_seg;

      cv::hconcat(croppedImg, pure_seg_mat, origin_seg);
      cv::hconcat(origin_seg, img_seg_show, origin_seg);
      Mat xyz_rgbl, final_compared;
      stereo_point_cloud stereoPointCloud;
      if (!grayImageR.empty())
      {
        stereoPointCloud.show_xyz_rgbl_plane_point_cloud_final(
            out_xyz_rgbl_cloud, xyz_rgbl);
        cv::vconcat(origin_seg, xyz_rgbl, final_compared);
        string finalPcdPath =
            config_.finalPcdDir + current_image_name_ + "_sub_cdt";
        savePcdfile_with_rgb_label(out_xyz_rgbl_cloud, finalPcdPath);
      }
      else
      {
        final_compared = origin_seg;
      }

      string finalPicPath =
          config_.finalPicDir + current_image_name_ + "_sub_cdt.jpg";
      imwrite(finalPicPath, final_compared);
    }

    // 保存点云到结果
    result.point_cloud = out_xyz_rgbl_cloud;

    return result;
  }
};

// 主程序
int main(int argc, char **argv)
{
  (void)argc; (void)argv;
  cout << "==================================================" << endl;
  cout << "  Offline Perception Debug Tool (Mode-Based)    " << endl;
  cout << "==================================================" << endl;

  // 读取配置
  OfflinePerceptionProcessor::Config config;
  // TODO: 从config.yaml读取配置
  config.infer_mode = 7;
  config.m_enable_debug_show = true;  // 启用调试输出（包括点云保存）
  // config.dsg_multi_model = "../models/dsg_multi_20260401_640x384.bin";
  bool ret_pcd_dir = true;

  // 默认路径
  // string input_dir = "/home/youfeng/debug/03/02/rosbag_LK-MR6P1US000107_camera_202603021453/stereo_output_rosbag_LK-MR6P1US000107_camera_202603021453_0/images/extracted_interval/brick/";
  // string input_dir = "/home/youfeng/debug/custom/0123/20260403/20260403/";
  // string input_dir = "/home/youfeng/debug/custom/0123/20260403/20260403/debug/";
  string input_dir = "/home/youfeng/debug/boluo/0286/20260408/";
  cout << "\nInput directory: " << input_dir << endl;

  // 构造最终结果输出目录
  const string mode_suffix =
      to_string(config.infer_mode) + "_" + to_string(config.erode_pixel);

  // 构造配置参数后缀
  string config_suffix = "";
  if (config.infer_mode == 6)
  {
    // 添加红砖颜色后处理参数
    if (config.enable_red_brick_refine_)
    {
      config_suffix += "_rbr" + to_string(config.red_brick_min_area_);
    }
    // 添加深度补全策略参数
    if (config.depth_inpainting_strategy_ > 0)
    {
      config_suffix += "_dip" + to_string(config.depth_inpainting_strategy_);
    }
  }

  if (config.infer_mode == 5)
  {
    config.finalPicDir = input_dir + "/cdt_mul_" + mode_suffix + "_0303_update_432/";
  }
  else if (config.infer_mode == 6)
  {
    config.finalPicDir = input_dir + "/cdt_sub_" + mode_suffix + config_suffix + "_0310_det_0.3_pc_432/";
  }
  else
  {
    config.finalPicDir = input_dir + "/dsg_multi_debug_" + mode_suffix + "_432/";
  }
  config.finalPcdDir =
      input_dir + "/dsg_pcd_debug_" + mode_suffix + "_" + std::to_string(config.erode_pixel) + "_432/"; // 设置最终PCD输出目录

  if (!config.finalPicDir.empty())
  {
    fs::create_directories(config.finalPicDir);
  }
  if (ret_pcd_dir && !config.finalPcdDir.empty())
  {
    fs::create_directories(config.finalPcdDir);
  }

  cout << "\n========== Configuration ==========" << endl;
  cout << "Inference mode: " << config.infer_mode << endl;
  cout << "Erode pixel: " << config.erode_pixel << endl;
  cout << "Area threshold: " << config.area_threshold << endl;
  cout << "Detection threshold: " << config.detection_threshold << endl;
  if (config.infer_mode == 6)
  {
    cout << "Red brick refine: " << (config.enable_red_brick_refine_ ? "enabled" : "disabled") << endl;
    if (config.enable_red_brick_refine_)
    {
      cout << "  Min area: " << config.red_brick_min_area_ << endl;
    }
    cout << "Depth inpainting strategy: " << config.depth_inpainting_strategy_ << endl;
  }
  cout << "===================================" << endl;

  // 初始化处理器
  OfflinePerceptionProcessor processor(config);
  if (!processor.init())
  {
    cerr << "\n[Error] Failed to initialize perception processor" << endl;
    return -1;
  }

  // 扫描输入文件
  vector<string> image_files;
  try
  {
    for (const auto &entry : fs::directory_iterator(input_dir))
    {
      if (!entry.is_regular_file())
        continue;

      string filename = entry.path().filename().string();
      string ext = entry.path().extension().string();

      // 支持常见图像格式
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
      {
        image_files.push_back(entry.path().string());
      }
    }
  }
  catch (const fs::filesystem_error &e)
  {
    cerr << "[Error] Cannot access input directory: " << e.what() << endl;
    return -1;
  }

  if (image_files.empty())
  {
    cout << "\n[Warning] No image files found in input directory." << endl;
    cout << "Supported formats: .jpg, .jpeg, .png, .bmp" << endl;
    return 0;
  }

  // 按文件名排序
  sort(image_files.begin(), image_files.end());

  cout << "\nFound " << image_files.size() << " image file(s)." << endl;

  // 处理每一张图像
  for (size_t i = 0; i < image_files.size(); i++)
  {
    const string &image_path = image_files[i];

    Mat full_img = imread(image_path);
    if (full_img.cols == 1280)
    {
      ret_pcd_dir = true;
    }

    if (full_img.empty())
    {
      cerr << "[Error] Cannot read image: " << image_path << endl;
      continue;
    }

    // 从完整图像中提取左右区域：
    // - 若为典型双目格式（宽度是高度的两倍，如
    // 1280x480），按水平中线切成左右两幅
    // - 否则认为是单目图像，只使用整幅作为 left_img
    int width = full_img.cols;
    int height = full_img.rows;
    int half_width = width / 2;

    Mat left_img = full_img.clone(); // 默认整幅作为左图
    Mat right_img;                   // 默认没有右图

    // 典型双目格式：左右拼接
    if (width == 1280 && height == 480)
    {
      left_img = full_img(Rect(0, 0, half_width, height)).clone();
      right_img = full_img(Rect(half_width, 0, half_width, height)).clone();
    }

    if (left_img.empty())
    {
      cerr << "[Error] Left image is empty after splitting: " << image_path
           << endl;
      continue;
    }

    cout << "\n[Image " << (i + 1) << "/" << image_files.size() << "] "
         << fs::path(image_path).filename().string() << endl;
    cout << "Full image size: " << full_img.size()
         << " -> Left/Right: " << left_img.size() << endl;

    // 获取原文件名（不含扩展名）
    string base_name = fs::path(image_path).stem().string();

    // 执行感知处理
    auto result = processor.process(left_img, right_img, i, base_name);
    //
    // // 生成输出文件名
    // string output_prefix = (fs::path(output_dir) / base_name).string();
    //
    // // 保存结果
    // if (!result.point_cloud.empty()) {
    //     // 保存PCD文件
    //     pcl::io::savePCDFileASCII(output_prefix + ".pcd",
    //     result.point_cloud); cout << "[Saved] " << output_prefix << ".pcd" <<
    //     endl;
    //
    //     // 保存三视图
    //     PointCloudVisualizer::saveThreeViews(result.point_cloud,
    //     output_prefix);
    //
    //     // 保存标签分布统计
    //     PointCloudVisualizer::saveLabelDistribution(result.point_cloud);
    // }
    //
    // // 保存深度图
    // if (!result.depth_map.empty()) {
    //     Mat depth_vis;
    //     normalize(result.depth_map, depth_vis, 0, 255, NORM_MINMAX, CV_8U);
    //     applyColorMap(depth_vis, depth_vis, COLORMAP_JET);
    //     imwrite(output_prefix + "_depth.jpg", depth_vis);
    //     cout << "[Saved] " << output_prefix << "_depth.jpg" << endl;
    // }
    //
    // // 保存标签图
    // if (!result.label_map.empty()) {
    //     Mat label_vis = result.label_map * 20;  // 放大显示
    //     imwrite(output_prefix + "_labels.png", label_vis);
    //     cout << "[Saved] " << output_prefix << "_labels.png" << endl;
    // }

    cout << "[Progress] " << (i + 1) << "/" << image_files.size()
         << " completed.\n"
         << endl;
  }

  cout << "\n==================================================" << endl;
  cout << "  All processing completed!                      " << endl;
  cout << "  Check output directory: " << config.finalPicDir << endl;
  cout << "==================================================" << endl;

  return 0;
}

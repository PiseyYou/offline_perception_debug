/**
 * @file offline_perception_debug.cpp
 * @brief 离线感知调试工具 - 按照ROS2 mode (0-6) 逻辑实现
 * @description 支持多种感知模式：depth-only, detection, segmentation,
 * multi-task等
 */

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
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

void convertIdToRGBOptimized(const cv::Mat &img_lab, cv::Mat &parsing_img)
{
  static const auto lut = getColorLookupTable();

  int rows = img_lab.rows;
  int cols = img_lab.cols;

  for (int i = 0; i < rows; ++i)
  {
    const uchar *row_ptr = img_lab.ptr<uchar>(i);
    cv::Vec3b *out_ptr = parsing_img.ptr<cv::Vec3b>(i);
    for (int j = 0; j < cols; ++j)
    {
      out_ptr[j] = lut[row_ptr[j]];
    }
  }
}

cv::Mat drawResultOptimized(cv::Mat &img_src, cv::Mat &img_lab,
                            std::vector<Detection> &dect_src,
                            cv::Mat &img_seg_show)
{

  // 1. 生成颜色图 (在较小的尺寸上操作)
  cv::Mat parsing_img(img_lab.size(), CV_8UC3);
  convertIdToRGBOptimized(img_lab, parsing_img);

  // 2. 将颜色图缩放到原图大小
  // 如果 img_lab 和 img_src 尺寸一致，此步会自动跳过或非常快
  if (parsing_img.size() != img_src.size())
  {
    cv::resize(parsing_img, parsing_img, img_src.size(), 0, 0,
               cv::INTER_NEAREST);
  }

  // 3. 图像融合 (Alpha Blending)
  // 建议先融合，这样绘制的框和文字才不会被半透明遮盖
  float alpha_f = 0.6f;
  cv::addWeighted(img_src, alpha_f, parsing_img, 1.0f - alpha_f, 0.0,
                  img_seg_show);

  // 4. 在融合后的图上绘制检测框
  for (const auto &det : dect_src)
  {
    cv::Rect rect_tmp(det.bbox.xmin, det.bbox.ymin,
                      (det.bbox.xmax - det.bbox.xmin),
                      (det.bbox.ymax - det.bbox.ymin));

    // 绘制矩形
    cv::rectangle(img_seg_show, rect_tmp, cv::Scalar(0, 0, 255), 2);

    // 通过 id+100 映射类别名称
    static const std::map<int, std::string> mul_map_class = {
        {0, "unla"},    {1, "back"},      {2, "gras"},      {3, "road"},
        {4, "dyna"},    {5, "stat"},      {6, "wall"},      {7, "vehi"},
        {8, "pole"},    {9, "impa"},      {10, "depr"},     {11, "bush"},
        {12, "limb"},   {13, "CES_arod"}, {14, "CES_stck"}, {15, "CES_pits"},
        {100, "pole"},  {101, "obst"},    {102, "fixo"},    {103, "car"},
        {104, "stat"},  {105, "dyna"},    {106, "chst"},    {107, "pers"}};
    int mapped_id = det.id + 100;
    auto it = mul_map_class.find(mapped_id);
    std::string obj_name = (it != mul_map_class.end()) ? it->second
                           : "id_" + std::to_string(det.id);
    std::string label = obj_name + ":" + cv::format("%.2f", det.score);

    cv::putText(img_seg_show, label,
                cv::Point(det.bbox.xmin,
                          std::max(static_cast<int>(det.bbox.ymin + 15), 15)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1,
                cv::LINE_AA);
  }

  return parsing_img;
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
    int infer_mode = 5;               // 推理模式 0-6
    // int erode_pixel = 205;            // 腐蚀像素
    int erode_pixel = 205;            // 腐蚀像素
    float area_threshold = 0.5;       // 区域阈值
    float detection_threshold = 0.2; // 检测阈值
    float m_area_threshold = 0.5;
    bool enable_height_filter_ = false;
    bool enabel_cdt = false;
    int frq_cdt = 5;
    bool m_enable_debug_show = true;

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
  };

  struct ProcessResult
  {
    pcl::PointCloud<pcl::PointXYZRGBL> point_cloud;
    Mat depth_map;
    Mat label_map;
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
  int current_frame_id_ = 0;       // 当前处理的帧ID
  string current_image_name_ = ""; // 当前处理的图像文件名（不含扩展名）

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

    // Step 4: 融合 (在 640x432 尺度下，使用裁剪后的深度图)
    auto fusion_start = chrono::high_resolution_clock::now();
    stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
        depth_432, result.label_map, dect_dst, croppedImg, xyz_rgbl_cloud,
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
      Mat depth_after_det = depthInpaintingByDetections(depth_cal, result.label_map, dect_dst);
      Mat depth_inpainted = depthInpaintingForObstacles(depth_after_det, result.label_map);
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
  // TODO: 从config.yaml读取配置（目前固定为 mode 6：Sub-task）
  config.infer_mode = 6;
  bool ret_pcd_dir = false;

  // 默认路径
  // string input_dir = "/home/youfeng/debug/03/03/userdata/rosbag_record/rosbag_LK-MR6P1US000107_camera_202603031104/stereo_output_rosbag_LK-MR6P1US000107_camera_202603031104_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/03/userdata/rosbag_record/rosbag_LK-MR6P1US000107_camera_202603031122/stereo_output_rosbag_LK-MR6P1US000107_camera_202603031122_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/03/userdata/rosbag_record/rosbag_LK-MR6P1US000107_camera_202603030959/stereo_output_rosbag_LK-MR6P1US000107_camera_202603030959_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/03/avoiding_people/rosbag_LK-MR6P1US000107_camera_202603031816/stereo_output_rosbag_LK-MR6P1US000107_camera_202603031816_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/04/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603041117/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603041117_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/04/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603041117/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603041117_0/images/extracted_interval/debug/";
  // string input_dir = "/home/youfeng/debug/03/04/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603041117/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603041117_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/04/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603041117/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603041117_0/images/extracted_interval/bottle/";
  // string input_dir = "/home/youfeng/debug/03/05/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603051439/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603051439_0_filtered_20260305_1439_to_20260305_1441/images/error/";
  // string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061618/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061618_0_filtered_20260306_1619_to_20260306_1621/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061439/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061439_0_filtered_20260306_1441_to_20260306_1443/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061551/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061551_0_filtered_20260306_1552_to_20260306_1554/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061404/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061404_0_filtered_20260306_1405_to_20260306_1408/images/no_point/";
  // string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061404/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061404_0_filtered_20260306_1405_to_20260306_1408/images/no_point/";
  // string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061404/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061404_0_filtered_20260306_1405_to_20260306_1408/images/no_point/";
  string input_dir = "/home/youfeng/debug/03/06/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061129/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061129_0/images/debug/error/";
  cout << "\nInput directory: " << input_dir << endl;

  // 构造最终结果输出目录
  const string mode_suffix =
      to_string(config.infer_mode) + "_" + to_string(config.erode_pixel);
  if (config.infer_mode == 5)
  {
    config.finalPicDir = input_dir + "/cdt_mul_" + mode_suffix + "_0303_update_432/";
  }
  else if (config.infer_mode == 6)
  {
    config.finalPicDir = input_dir + "/cdt_sub_" + mode_suffix + "_0304_det_0.2_pc_432/";
  }
  config.finalPcdDir =
      input_dir + "/pcd_" + mode_suffix + "_0304_432/"; // 设置最终PCD输出目录

  fs::create_directories(config.finalPicDir);
  if (ret_pcd_dir)
  {
    fs::create_directories(config.finalPcdDir);
  }

  cout << "\n========== Configuration ==========" << endl;
  cout << "Inference mode: " << config.infer_mode << endl;
  cout << "Erode pixel: " << config.erode_pixel << endl;
  cout << "Area threshold: " << config.area_threshold << endl;
  cout << "Detection threshold: " << config.detection_threshold << endl;
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

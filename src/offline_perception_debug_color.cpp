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


// 离线感知处理器 - 支持多种模式
class OfflinePerceptionProcessor {
public:
  // 配置参数
  struct Config {
    int infer_mode = 6;               // 推理模式 0-6
    int erode_pixel = 205;            // 腐蚀像素
    float area_threshold = 0.5;       // 区域阈值
    float detection_threshold = 0.51; // 检测阈值
    float m_area_threshold = 0.5;
    bool enable_height_filter_ = false;
    bool enabel_cdt = false;
    int frq_cdt = 5;
    bool m_enable_debug_show = true;
    bool use_color_judge = true; // true: 基于颜色判断label; false: 基于DL模型

    string finalPicDir = ""; // 最终图片输出目录
    string finalPcdDir = ""; // 最终PCD输出目录

    string cdt_model = "../models/cdt_20251125_640x384.bin";
    string det_model = "../models/det_20241106_640x480.bin";
    string seg_model = "../models/seg_20250421_640x384.bin";
    string multi_model = "../models/mul_20250918_640x384.bin";
    string cs_model = "../models/cqr_20250821_640x384_yolov8n.bin";
    // string sub_model = "../models/sub_20260105_640x384.bin";
    string sub_model = "../models/sub_20260211_640x384.bin";
  };

  struct ProcessResult {
    pcl::PointCloud<pcl::PointXYZRGBL> point_cloud;
    Mat depth_map;
    Mat label_map;
    double process_time_ms;
    int frame_id;
    string mode_name;
  };

  OfflinePerceptionProcessor(const Config &cfg) : config_(cfg) {}

  bool init() {
    cout << "\n========== Initializing Perception Modules ==========" << endl;

    // 初始化立体匹配
    stereo_multi_match.stereo_multi_param_init();
    cout << "[✓] Stereo matcher initialized" << endl;

    cdtPerception_.perception_init(config_.cdt_model.c_str());
    cout << "[✓] Cdt-task model initialized: " << config_.cdt_model << endl;

    // 根据模式初始化对应的感知模块
    if (config_.infer_mode == 0) {
      cout << "[Mode 0] Disable mode - no processing" << endl;
    } else if (config_.infer_mode == 1) {
      cout << "[Mode 1] Only depth mode - no DL models needed" << endl;
    } else if (config_.infer_mode == 2) {
      detPerception_.perception_init(config_.det_model.c_str(),
                                     config_.detection_threshold);
      cout << "[✓] Detection model initialized: " << config_.det_model << endl;
    } else if (config_.infer_mode == 3) {
      segPerception_.perception_init(config_.seg_model.c_str());
      cout << "[✓] Segmentation model initialized: " << config_.seg_model
           << endl;
    } else if (config_.infer_mode == 4) {
      qrCsPerception_.perception_init(config_.cs_model.c_str());
      cout << "[✓] CS/QR model initialized: " << config_.cs_model << endl;
    } else if (config_.infer_mode == 5) {
      multiPerception_.perception_init(config_.multi_model.c_str());
      cout << "[✓] Multi-task model initialized: " << config_.multi_model
           << endl;
    } else if (config_.infer_mode == 6) {
      mulSubPerception.perception_init(config_.sub_model.c_str());
      cout << "[✓] Sub-task model initialized: " << config_.sub_model << endl;
    }

    cout << "===================================================\n" << endl;
    return true;
  }

  ProcessResult process(const Mat &left_img, const Mat &right_img, int frame_id,
                        const string &image_name = "") {
    auto start_time = chrono::high_resolution_clock::now();

    ProcessResult result;
    result.frame_id = frame_id;
    current_frame_id_ = frame_id;     // 保存当前帧ID
    current_image_name_ = image_name; // 保存当前图像文件名

    if (left_img.empty()) {
      cerr << "[Error] process() received empty left image for frame "
           << frame_id << ", image name: " << image_name << endl;
      return result;
    }

    // 转换为灰度图用于立体匹配（右图可能为空）
    Mat grayImageLeft, grayImageRight;
    cvtColor(left_img, grayImageLeft, COLOR_BGR2GRAY);
    if (!right_img.empty()) {
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
    switch (config_.infer_mode) {
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

  // Mode 0: Disable - 不进行任何处理
  ProcessResult processMode0(const Mat &left, const Mat &right,
                             const Mat &grayImageL, const Mat &grayImageR) {
    ProcessResult result;
    result.mode_name = "Disable";
    cout << "[Mode 0] Disable - no processing" << endl;
    return result;
  }

  // Mode 1: Only Depth - 仅深度，无DL模型
  ProcessResult processMode1(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR) {
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
                             Mat grayImageR) {
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
                             Mat grayImageR) {
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
                             Mat grayImageR) {
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

  // 基于颜色判断生成label图 (亮度分层 + RGB分量)
  // label=1: 暗区/过亮区(天花板、玻璃、灯光等)
  // label=2: 绿色(草地) — G分量主导
  // label=5: 灰白色(墙壁/隔板) — 低色差中等亮度
  // label=3: 其余道路
  cv::Mat generateColorLabel(const cv::Mat &bgr_img) {
    cv::Mat label_map(bgr_img.rows, bgr_img.cols, CV_8UC1, cv::Scalar(1));

    // 邻域均值滤波，消除单像素噪声
    cv::Mat blurred;
    cv::blur(bgr_img, blurred, cv::Size(25, 25));

    for (int i = 0; i < blurred.rows; ++i) {
      const cv::Vec3b *ptr = blurred.ptr<cv::Vec3b>(i);
      uchar *label_ptr = label_map.ptr<uchar>(i);
      for (int j = 0; j < blurred.cols; ++j) {
        int b = ptr[j][0];
        int g = ptr[j][1];
        int r = ptr[j][2];
        int sum = r + g + b;
        int max_val = std::max({r, g, b});
        int min_val = std::min({r, g, b});
        int range = max_val - min_val;

        // Layer 1: 暗区 -> label=1 (玻璃、远处暗区、天花板黑条)
        if (sum < 150) {
          label_ptr[j] = 1;
          continue;
        }
        // Layer 2: 过亮区 -> label=1 (灯光、高光反射)
        if (sum > 650) {
          label_ptr[j] = 1;
          continue;
        }

        // --- 中间亮度范围内做细分 ---

        // 灰白色: 低色差 + 中等以上亮度 -> label=5 (墙壁/隔板)
        // 放宽: range < 50 (was 35), sum > 200 (was 240)
        if (range < 50 && sum > 200) {
          label_ptr[j] = 5;
        }
        // 绿色: G分量相对主导 -> label=2 (草地)
        // 放宽: G不需要严格大于R和B，允许接近 (G >= R-5 && G >= B-5)
        else if (g > 45 && g >= r - 5 && g >= b - 5 && g == max_val) {
          label_ptr[j] = 2;
        }
        // 其余中间亮度 -> label=3 (道路)
        else {
          label_ptr[j] = 3;
        }
      }
    }

    // 形态学处理 — 闭运算填孔 + 开运算去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));

    cv::Mat mask2 = (label_map == 2);
    cv::morphologyEx(mask2, mask2, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask2, mask2, cv::MORPH_OPEN, kernel);

    cv::Mat mask5 = (label_map == 5);
    cv::morphologyEx(mask5, mask5, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask5, mask5, cv::MORPH_OPEN, kernel);

    cv::Mat mask1 = (label_map == 1);
    cv::morphologyEx(mask1, mask1, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask1, mask1, cv::MORPH_OPEN, kernel);

    // 重建 label_map: 优先级 1 > 5 > 2 > 3
    label_map.setTo(3);
    label_map.setTo(2, mask2);
    label_map.setTo(5, mask5);
    label_map.setTo(1, mask1);

    return label_map;
  }

  // Mode 5: Multi-task recognition - 多任务识别 (分割+检测)
  ProcessResult processMode5(const Mat &left, const Mat &right, Mat grayImageL,
                             Mat grayImageR) {
    ProcessResult result;
    result.mode_name = "MultiTask";

    cout << "[Mode 5] Multi-task recognition (seg + det)" << endl;

    // Step 1: 获取label图
    auto infer_start = chrono::high_resolution_clock::now();

    cv::Rect cropRegion(0, 0, left.cols, 432);
    cv::Mat croppedImg = left(cropRegion);
    // 对灰度图也进行裁剪，保证后续深度计算在 640x432 尺度下进行，免去多余计算
    cv::Mat grayL_432 = grayImageL(cropRegion);
    cv::Mat grayR_432 = grayImageR(cropRegion);

    cv::Mat label_432;
    std::vector<Detection> dect_dst;

    if (config_.use_color_judge) {
      // ===== 基于颜色判断生成label (不使用DL模型) =====
      cout << "[Color Mode] Using color-based label generation" << endl;
      label_432 = generateColorLabel(croppedImg); // 直接在640x432上做颜色判断
      // 颜色模式下无检测框
    } else {
      // ===== 基于DL模型推理 =====
      Mat resizeImg = cv::Mat::zeros(384, 640, croppedImg.type());
      cv::resize(croppedImg, resizeImg, cv::Size(640, 384));

      cv::Mat dst_label(resizeImg.rows, resizeImg.cols, CV_8UC1);
      std::vector<Detection> detections, ct_dect_src;
      cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
      Mat lab_out, lab_temp;

      Rect cdt_rect;
      if (config_.enabel_cdt) {
        ct_dect_src.clear();
        cdtPerception_.perception_process_bgr(croppedImg, ct_dect_src);
        cdt_rect = get_cdt_rect(ct_dect_src);
        cout << "cdt_rect: " << cdt_rect << endl;
        cout << "cdt_rect.area: " << cdt_rect.area() << endl;
      }

      // 网络在 384 下推理
      multiPerception_.perception_process_bgr_no_argmax_erode_mul(
          resizeImg, detections, img_label, config_.erode_pixel);
      filterLabelDect(img_label, detections, dst_label, dect_dst);

      // 将标签图还原回 432
      cv::resize(dst_label, label_432, cv::Size(640, 432), 0, 0,
                 cv::INTER_NEAREST);

      // 将 384 尺度下获取的检测框还原回 432
      for (auto &det : dect_dst) {
        det.bbox.ymin =
            std::max(0, static_cast<int>(det.bbox.ymin * (432.0f / 384.0f)));
        det.bbox.ymax =
            std::min(432, static_cast<int>(det.bbox.ymax * (432.0f / 384.0f)));
      }

      // 将 cdt 遮罩施加在放大后的 label_432 上
      if (config_.enabel_cdt && cdt_rect.area() > 0) {
        label_432(cdt_rect) = -1;
      }
    }

    auto infer_end = chrono::high_resolution_clock::now();
    cout << "[Step 1/4] "
         << (config_.use_color_judge ? "Color-based" : "Multi-task inference")
         << " done: "
         << chrono::duration<double, milli>(infer_end - infer_start).count()
         << " ms" << endl;

    // Step 2: 标签后处理 (简化版，跳过复杂的label processing)
    result.label_map = label_432.clone(); // 此后点云融合只使用 640x432 的 label
    cout << "[Step 2/4] Label processing done (simplified)" << endl;

    // Step 3: 深度计算 (严格在 640x432 尺度下)
    auto depth_start = chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud, out_xyz_rgbl_cloud;
    Mat disparity =
        stereo_multi_match.stereo_multi_process_depth(grayL_432, grayR_432);
    Mat depth_cal = stereo_multi_match.stereo_multi_process_filter(
        disparity, result.label_map, config_.enable_height_filter_);

    auto depth_end = chrono::high_resolution_clock::now();
    cout << "[Step 3/4] Depth computation done: "
         << chrono::duration<double, milli>(depth_end - depth_start).count()
         << " ms" << endl;

    // Step 4: 融合
    auto fusion_start = chrono::high_resolution_clock::now();
    // 全部送入 640x432 规格的计算参数
    stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
        depth_cal, result.label_map, dect_dst, croppedImg, xyz_rgbl_cloud,
        out_xyz_rgbl_cloud);

    if (config_.m_enable_debug_show) {
      Mat img_seg_show;
      // 因为绘图使用的 dect_dst 已经是 432 高度的比例了，所以需要绘制在
      // croppedImg(640x432) 以及 label_432(640x432) 上
      Mat pure_seg_mat =
          drawResultOptimized(croppedImg, label_432, dect_dst, img_seg_show);

      Mat origin_seg;
      // 不需要额外再 resize，pure_seg_mat 本身已经是 432 了
      // cv::resize(pure_seg_mat, pure_seg_mat, Size(640, 432));
      // cv::resize(img_seg_show, img_seg_show, Size(640, 432));

      cv::hconcat(croppedImg, pure_seg_mat, origin_seg); // 拼接图片1和图片2
      cv::hconcat(origin_seg, img_seg_show, origin_seg); // 拼接图片1和图片2
      Mat xyz_rgbl, final_compared;
      stereo_point_cloud stereoPointCloud;
      stereoPointCloud.show_xyz_rgbl_plane_point_cloud_final(out_xyz_rgbl_cloud,
                                                             xyz_rgbl);
      cv::vconcat(origin_seg, xyz_rgbl,
                  final_compared); // 再将图片3拼接到结果中

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
                             Mat grayImageR) {
    ProcessResult result;
    result.mode_name = "SubTask";

    cout << "[Mode 6] Sub-task recognition" << endl;

    // Step 1: 获取label图
    auto infer_start = chrono::high_resolution_clock::now();

    cv::Mat croppedImg;
    if (left.rows == 480) {
      cv::Rect cropRegion(0, 0, left.cols, 432);
      croppedImg = left(cropRegion);
    } else if (left.rows == 384) {
      croppedImg = left;
    } else {
      croppedImg = left;
    }

    cv::Mat label_432;
    std::vector<Detection> dect_dst;

    if (config_.use_color_judge) {
      // ===== 基于颜色判断生成label (不使用DL模型) =====
      cout << "[Color Mode] Using color-based label generation" << endl;
      label_432 = generateColorLabel(croppedImg); // 640x432
    } else {
      // ===== 基于DL模型推理 =====
      cv::Mat resizeImg = cv::Mat::zeros(384, 640, croppedImg.type());
      cv::resize(croppedImg, resizeImg, cv::Size(640, 384));

      cv::Mat dst_label(resizeImg.rows, resizeImg.cols, CV_8UC1);
      std::vector<Detection> detections, ct_dect_src;
      cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
      Mat lab_out, lab_temp;

      Rect cdt_rect;

      mulSubPerception.perception_process_bgr_no_argmax_erode(
          resizeImg, detections, img_label, lab_out, config_.erode_pixel);
      filterLabelDect(lab_out, detections, dst_label, dect_dst);

      // 还原 label 至真实尺寸
      if (dst_label.rows == 384 && croppedImg.rows == 432) {
        cv::resize(dst_label, label_432, cv::Size(640, 432), 0, 0,
                   cv::INTER_NEAREST);
        for (auto &det : dect_dst) {
          det.bbox.ymin =
              std::max(0, static_cast<int>(det.bbox.ymin * (432.0f / 384.0f)));
          det.bbox.ymax =
              std::min(432, static_cast<int>(det.bbox.ymax * (432.0f / 384.0f)));
        }
      } else {
        label_432 = dst_label.clone();
      }

      if (cdt_rect.area() > 0) {
        label_432(cdt_rect) = -1;
      }
    }

    auto infer_end = chrono::high_resolution_clock::now();
    cout << "[Step 1/3] "
         << (config_.use_color_judge ? "Color-based" : "Sub-task inference")
         << " done: "
         << chrono::duration<double, milli>(infer_end - infer_start).count()
         << " ms" << endl;

    result.label_map = label_432.clone();

    // 可选的点云输出
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud, out_xyz_rgbl_cloud;

    auto fusion_start = chrono::high_resolution_clock::now();
    if (!grayImageR.empty()) {
      // Step 2: 深度计算 — 使用原始 640x480 灰度图，不裁剪
      auto depth_start = chrono::high_resolution_clock::now();

      Mat disparity =
          stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);

      // 将 label_432 pad 到 640x480 送入 filter（底部48行填0，filter内部不会对label=0生成mask）
      cv::Mat label_480 = cv::Mat::zeros(grayImageL.rows, grayImageL.cols, CV_8UC1);
      label_432.copyTo(label_480(cv::Rect(0, 0, label_432.cols, label_432.rows)));

      Mat depth_480 = stereo_multi_match.stereo_multi_process_filter(
          disparity, label_480, config_.enable_height_filter_);

      // 裁剪深度图到 640x432，与 label / croppedImg 对齐
      cv::Mat depth_432 = depth_480(cv::Rect(0, 0, 640, 432)).clone();

      auto depth_end = chrono::high_resolution_clock::now();
      cout << "[Step 2/3] Depth computation done: "
           << chrono::duration<double, milli>(depth_end - depth_start).count()
           << " ms" << endl;

      // Step 3: 融合 (所有元件统一 640x432)
      stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
          depth_432, result.label_map, dect_dst, croppedImg, xyz_rgbl_cloud,
          out_xyz_rgbl_cloud);
    };

    if (config_.m_enable_debug_show) {
      Mat img_seg_show;
      Mat pure_seg_mat = drawResultOptimized(croppedImg, result.label_map,
                                             dect_dst, img_seg_show);
      Mat origin_seg;
      cv::hconcat(croppedImg, pure_seg_mat, origin_seg);
      cv::hconcat(origin_seg, img_seg_show, origin_seg);

      Mat xyz_rgbl, final_compared;
      stereo_point_cloud stereoPointCloud;
      if (!grayImageR.empty()) {
        stereoPointCloud.show_xyz_rgbl_plane_point_cloud_final(
            out_xyz_rgbl_cloud, xyz_rgbl);
        cv::vconcat(origin_seg, xyz_rgbl, final_compared);
        string finalPcdPath =
            config_.finalPcdDir + current_image_name_ + "_sub_cdt";
        savePcdfile_with_rgb_label(out_xyz_rgbl_cloud, finalPcdPath);
      } else {
        final_compared = origin_seg;
      }

      string finalPicPath =
          config_.finalPicDir + current_image_name_ + "_sub_cdt.jpg";
      imwrite(finalPicPath, final_compared);
    }
    auto fusion_end = chrono::high_resolution_clock::now();
    cout << "[Step 3/3] Fusion done: "
         << chrono::duration<double, milli>(fusion_end - fusion_start).count()
         << " ms" << endl;

    return result;
  }
};

// 主程序
int main(int argc, char **argv) {
  cout << "==================================================" << endl;
  cout << "  Offline Perception Debug Tool (Mode-Based)    " << endl;
  cout << "==================================================" << endl;

  // 读取配置
  OfflinePerceptionProcessor::Config config;
  // TODO: 从config.yaml读取配置（目前固定为 mode 6：Sub-task）
  config.infer_mode = 6;
  bool ret_pcd_dir = false;

  // 默认路径
  // string input_dir =
  // "/home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug/data/test/";
  // string input_dir =
  // "/home/youfeng/debug/select/CES/indoor/rosbag_MR1P1251US0000844_navigation_202601031719/left_images_20260103_171930/";
  // string input_dir =
  // "/home/youfeng/debug/01/0107/rosbag_MR1P1251US0000844_navigation_202601061805/stereo_output_rosbag_MR1P1251US0000844_navigation_202601061805_0/images/stereo_interval/";
  // string input_dir =
  // "/home/youfeng/debug/01/0109/rosbag_MR1P1251US0008162_camera_202601091914/stereo_output_rosbag_MR1P1251US0008162_camera_202601091914_0/images/";
  // string input_dir =
  // "/home/youfeng/debug/01/0109/rosbag_MR1P1251US0008162_camera_202601091918/stereo_output_rosbag_MR1P1251US0008162_camera_202601091918_0/images/";
  // string input_dir =
  // "/home/youfeng/debug/01/0113/rosbag_MR1P1251US0000126_navigation_202601122259/stereo_output_rosbag_MR1P1251US0000126_navigation_202601122259_0/images/";
  // string input_dir =
  // "/home/youfeng/debug/2025/10/rosbag_MR1P1251US0000266_navigation_202510301430/left_images_20251030_143020/";
  // string input_dir =
  // "/home/youfeng/debug/01/0120/test/ros_n_c/rosbag_record/rosbag_LK-MR2P1US000016_camera_202601201013/stereo_output_rosbag_LK-MR2P1US000016_camera_202601201013_0/images/";
  // string input_dir =
  // "/home/youfeng/debug/01/0120/test/ros_n_c/rosbag_record/rosbag_LK-MR2P1US000016_camera_202601201013/stereo_output_rosbag_LK-MR2P1US000016_camera_202601201013_0/images/";
  // string input_dir =
  // "/home/youfeng/debug/01/0120/test/rosbag_LK-MR6P1US000103_navigation_202601201140/stereo_output_rosbag_LK-MR6P1US000103_navigation_202601201140_0/images/";
  // string input_dir =
  // "/home/youfeng/debug/01/0120/test/rosbag_LK-MR6P1US000107_navigation_202601201151/stereo_output_rosbag_LK-MR6P1US000107_navigation_202601201151_0/images/";
  // string input_dir = "/home/youfeng/debug/custom/6506/0210/stereo/pic/";
  // string input_dir =
  // "/home/youfeng/debug/02/10/rosbag_LK-MR6P1US000111_navigation_202602101504/stereo_output_rosbag_LK-MR6P1US000111_navigation_202602101504_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/custom/0521/0211/stereo/pic/";
  // string input_dir =
  // "/home/youfeng/debug/02/10/rosbag_LK-MR6P1US000111_navigation_202602101504/stereo_output_rosbag_LK-MR6P1US000111_navigation_202602101504_0/images/extracted_interval/error_stereo/";
  // string input_dir =
  // "/home/youfeng/debug/02/11/rosbag_LK-MR6P1US000107_camera_202602111630/stereo_output_rosbag_LK-MR6P1US000107_camera_202602111630_0_filtered_20260211_1632_to_20260211_1632/images/";
  // string input_dir = "/home/youfeng/debug/custom/6506/0210/stereo/pic/";
  // string input_dir =
  // "/home/youfeng/debug/02/13/userdata/rosbag_record/rosbag_LK-MR6P1US000107_camera_202602131009/stereo_output_rosbag_LK-MR6P1US000107_camera_202602131009_0/images/extracted_interval/";
  // string input_dir =
  // "/home/youfeng/debug/02/13/userdata/rosbag_record/rosbag_LK-MR6P1US000107_camera_202602130944/stereo_output_rosbag_LK-MR6P1US000107_camera_202602130944_0/images/extracted_interval/";
  // string input_dir = "/home/youfeng/debug/custom/6506/0212/stereo/pic/";
  // string input_dir =
  // "/home/youfeng/debug/02/24/rosbag_LK-MR601CN000000_navigation_202602241759/stereo_output_rosbag_LK-MR601CN000000_navigation_202602241759_0/images/extracted_interval/";
  // string input_dir =
  // "/home/youfeng/debug/02/24/rosbag_LK-MR601CN000000_navigation_202602241759/stereo_output_rosbag_LK-MR601CN000000_navigation_202602241759_0/select_pic/";
  // string input_dir =
  // "/home/youfeng/debug/02/24/rosbag_LK-MR601CN000000_navigation_202602241759/stereo_output_rosbag_LK-MR601CN000000_navigation_202602241759_0/select_pic/";
  string input_dir = "/home/youfeng/debug/02/24/"
                     "rosbag_LK-MR601CN000000_navigation_202602241759/"
                     "stereo_output_rosbag_LK-MR601CN000000_navigation_"
                     "202602241759_0/images/extracted_interval/";
  cout << "\nInput directory: " << input_dir << endl;

  // 构造最终结果输出目录
  const string mode_suffix =
      to_string(config.infer_mode) + "_" + to_string(config.erode_pixel);
  if (config.infer_mode == 5) {
    config.finalPicDir = input_dir + "/cdt_mul_" + mode_suffix + "/";
  } else if (config.infer_mode == 6) {
    config.finalPicDir = input_dir + "/cdt_sub_" + mode_suffix + "_0225_432_color2/";
  }
  config.finalPcdDir =
      input_dir + "/pcd_" + mode_suffix + "_0225_432_color2/"; // 设置最终PCD输出目录

  fs::create_directories(config.finalPicDir);
  if (ret_pcd_dir) {
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
  if (!processor.init()) {
    cerr << "\n[Error] Failed to initialize perception processor" << endl;
    return -1;
  }

  // 扫描输入文件
  vector<string> image_files;
  try {
    for (const auto &entry : fs::directory_iterator(input_dir)) {
      if (!entry.is_regular_file())
        continue;

      string filename = entry.path().filename().string();
      string ext = entry.path().extension().string();

      // 支持常见图像格式
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        image_files.push_back(entry.path().string());
      }
    }
  } catch (const fs::filesystem_error &e) {
    cerr << "[Error] Cannot access input directory: " << e.what() << endl;
    return -1;
  }

  if (image_files.empty()) {
    cout << "\n[Warning] No image files found in input directory." << endl;
    cout << "Supported formats: .jpg, .jpeg, .png, .bmp" << endl;
    return 0;
  }

  // 按文件名排序
  sort(image_files.begin(), image_files.end());

  cout << "\nFound " << image_files.size() << " image file(s)." << endl;

  // 处理每一张图像
  for (size_t i = 0; i < image_files.size(); i++) {
    const string &image_path = image_files[i];

    Mat full_img = imread(image_path);
    if (full_img.cols == 1280) {
      ret_pcd_dir = true;
    }

    if (full_img.empty()) {
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
    if (width == 1280 && height == 480) {
      left_img = full_img(Rect(0, 0, half_width, height)).clone();
      right_img = full_img(Rect(half_width, 0, half_width, height)).clone();
    }

    if (left_img.empty()) {
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

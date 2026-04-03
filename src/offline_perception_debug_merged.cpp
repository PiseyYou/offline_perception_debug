/**
 * @file offline_perception_debug_merged.cpp
 * @brief 离线感知调试工具 - 融合 432px 和 384px 高度逻辑
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

#include "cdt_perception.h"
#include "det_perception.h"
#include "multi_sub_perception.h"
#include "offline_utils.hpp"
#include "qr_cs_perception.h"
#include "seg_perception.h"
#include "dsg_perception.h"
#include "stereo_multi_match.h"
#include "stereo_point_cloud_rgbl.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// 统计单通道 label 中不同 ID 的数量
void countLabelStatistics(const cv::Mat &label_map) {
  if (label_map.empty()) return;

  std::map<int, int> id_counts;
  int total_pixels = label_map.total();

  for (int i = 0; i < label_map.rows; ++i) {
    const uchar *row_ptr = label_map.ptr<uchar>(i);
    for (int j = 0; j < label_map.cols; ++j) {
      int id = static_cast<int>(row_ptr[j]);
      id_counts[id]++;
    }
  }

  std::cout << "Label Statistics: " << label_map.cols << " x " << label_map.rows
            << " (Total: " << total_pixels << " pixels)" << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;
  std::cout << std::setw(10) << "ID"
            << std::setw(15) << "Pixel Count"
            << std::setw(12) << "Percentage" << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;

  for (const auto &pair : id_counts) {
    int id = pair.first;
    int count = pair.second;
    double percentage = (static_cast<double>(count) / total_pixels) * 100.0;

    std::cout << std::setw(10) << id
              << std::setw(15) << count
              << std::setw(11) << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
  }
  std::cout << "===================================================" << std::endl;
}

class OfflinePerceptionProcessor {
public:
  struct Config {
    int infer_mode = 7;
    int erode_pixel = 205;
    float detection_threshold = 0.3;
    bool enable_height_filter_ = false;
    bool enabel_cdt = false;
    bool m_enable_debug_show = true;
    bool use_color_judge = false;
    int depth_inpainting_strategy = 0;

    string finalPicDir = "";
    string finalPcdDir = "";

    string cdt_model = "../models/cdt_20251125_640x384.bin";
    string det_model = "../models/det_20241106_640x480.bin";
    string seg_model = "../models/seg_20250421_640x384.bin";
    string multi_model = "../models/mul_20250918_640x384.bin";
    string cs_model = "../models/cqr_20250821_640x384_yolov8n.bin";
    string sub_model = "../models/sub_20260303_640x384.bin";
  };

  OfflinePerceptionProcessor(const Config &cfg) : config_(cfg) {}

  bool init() {
    stereo_multi_match.stereo_multi_param_init();
    cdtPerception_.perception_init(config_.cdt_model.c_str());

    if (config_.infer_mode == 2) {
      detPerception_.perception_init(config_.det_model.c_str(), config_.detection_threshold);
    } else if (config_.infer_mode == 3) {
      segPerception_.perception_init(config_.seg_model.c_str());
    } else if (config_.infer_mode == 4) {
      qrCsPerception_.perception_init(config_.cs_model.c_str());
    } else if (config_.infer_mode == 5) {
      multiPerception_.perception_init(config_.multi_model.c_str());
    } else if (config_.infer_mode == 6) {
      mulSubPerception.perception_init(config_.sub_model.c_str());
    } else if (config_.infer_mode == 7) {
      dsgPerception_.perception_init(config_.multi_model.c_str());
    }
    return true;
  }

  void process(const Mat &left_img, const Mat &right_img, int frame_id, const string &base_name) {
    (void)frame_id;
    current_image_name_ = base_name;
    Mat grayL, grayR;
    cvtColor(left_img, grayL, COLOR_BGR2GRAY);
    if (!right_img.empty()) cvtColor(right_img, grayR, COLOR_BGR2GRAY);

    // 1. 裁剪成 640x432
    cv::Rect cropRegion(0, 0, left_img.cols, 432);
    cv::Mat croppedImg = left_img(cropRegion);

    // 2. resize 到 640x384 进行推理
    cv::Mat resizedForSeg;
    cv::resize(croppedImg, resizedForSeg, cv::Size(640, 384));

    vector<Detection> detections, dect_dst, cdt_src;
    Mat img_label = Mat::zeros(384, 640, CV_8UC1) + 2;
    Mat lab_out = Mat::zeros(384, 640, CV_8UC1);
    Mat dst_label;

    if (config_.infer_mode == 6) {
      dst_label = Mat::zeros(384, 640, CV_8UC1);
      mulSubPerception.perception_process_bgr_no_argmax_erode(resizedForSeg, detections, img_label, lab_out, config_.erode_pixel);
      filterLabelDect(lab_out, detections, dst_label, dect_dst);
    } else if (config_.infer_mode == 7) {
      if (config_.enabel_cdt) {
        cdtPerception_.perception_process_bgr(resizedForSeg, cdt_src);
      }
      // 设置原始尺寸用于坐标对齐
      dsgPerception_.ori_height = 384;
      dsgPerception_.ori_width = 640;

      // 3. 推理得到 640x384 的结果
      dsgPerception_.process_infer_match(resizedForSeg, lab_out);

      // 4. 将分割结果 640x384 resize 回 640x432
      cv::resize(lab_out, dst_label, cv::Size(640, 432), 0, 0, cv::INTER_NEAREST);

      cv::Rect cdt_rect = get_cdt_rect(cdt_src);
      if (config_.enabel_cdt && cdt_rect.area() > 0) {
        dst_label(cdt_rect) = -1;
      }
    }

    // 输出类别统计信息
    countLabelStatistics(dst_label);
    std::cout << "\n[Frame Summary] Processing complete for: " << base_name << "\n" << std::endl;

    // 最终用于融合的 label (640x432)
    Mat label_final = dst_label.clone();
    int target_h = 432;

    if (config_.infer_mode == 6) {
        resize(dst_label, label_final, Size(640, 432), 0, 0, INTER_NEAREST);
        for (auto &det : dect_dst) {
            det.bbox.ymin *= (float)target_h / 384.0f;
            det.bbox.ymax *= (float)target_h / 384.0f;
        }
    }

    if (!grayR.empty()) {
      Mat disparity = stereo_multi_match.stereo_multi_process_depth(grayL, grayR);

      // Pad label to match grayL.rows for filter if needed
      Mat label_pad = Mat::zeros(grayL.rows, grayL.cols, CV_8UC1) + 2;
      label_final.copyTo(label_pad(Rect(0, 0, 640, target_h)));

      Mat depth_pad = stereo_multi_match.stereo_multi_process_filter(disparity, label_pad, config_.enable_height_filter_);
      Mat depth_final = depth_pad(Rect(0, 0, 640, target_h)).clone();

      pcl::PointCloud<pcl::PointXYZRGBL> cloud, out_cloud;
      stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(depth_final, label_final, dect_dst, croppedImg, cloud, out_cloud);

      if (config_.m_enable_debug_show) {
        Mat img_show, pure_seg;
        pure_seg = drawResultOptimized(croppedImg, label_final, dect_dst, img_show);

        Mat xyz_rgbl, final_compared;
        stereo_point_cloud stereoPointCloud;
        stereoPointCloud.show_xyz_rgbl_plane_point_cloud_final(out_cloud, xyz_rgbl);

        Mat top_row;
        hconcat(croppedImg, pure_seg, top_row);
        hconcat(top_row, img_show, top_row);
        vconcat(top_row, xyz_rgbl, final_compared);

        imwrite(config_.finalPicDir + base_name + "_merged.jpg", final_compared);
        savePcdfile_with_rgb_label(out_cloud, config_.finalPcdDir + base_name + "_merged");
      }
    }
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
  string current_image_name_;
};

int main() {
  string input_dir = "/home/youfeng/debug/custom/0123/0330/20260328/";
  if (!fs::exists(input_dir)) {
    cout << "Input directory does not exist: " << input_dir << endl;
    return 0;
  }

  OfflinePerceptionProcessor::Config cfg;
  cfg.finalPicDir = input_dir + "debug_pic/";
  cfg.finalPcdDir = input_dir + "debug_pcd/";
  try {
    if (!fs::exists(cfg.finalPicDir)) fs::create_directories(cfg.finalPicDir);
    if (!fs::exists(cfg.finalPcdDir)) fs::create_directories(cfg.finalPcdDir);
  } catch (const fs::filesystem_error& e) {
    cerr << "Error creating directories: " << e.what() << endl;
    // Fallback: Use current directory if absolute path fails
    cfg.finalPicDir = "./debug_pic/";
    cfg.finalPcdDir = "./debug_pcd/";
    fs::create_directories(cfg.finalPicDir);
    fs::create_directories(cfg.finalPcdDir);
  }

  OfflinePerceptionProcessor processor(cfg);
  if (!processor.init()) return -1;

  for (const auto &entry : fs::directory_iterator(input_dir)) {
    if (entry.path().extension() == ".jpg") {
      Mat img = imread(entry.path().string());
      if (img.empty()) continue;
      if (img.cols == 1280) {
        Mat left = img(Rect(0, 0, 640, img.rows)).clone();
        Mat right = img(Rect(640, 0, 640, img.rows)).clone();
        processor.process(left, right, 0, entry.path().stem().string());
      }
    }
  }
  return 0;
}

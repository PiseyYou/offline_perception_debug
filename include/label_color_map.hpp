#pragma once

#include <map>
#include <opencv2/opencv.hpp>

// 统一的语义标签颜色映射（BGR）
// 所有模块（2D 分割可视化、点云俯视图等）都应通过这里获取颜色，避免重复定义。
namespace label_color {

inline const std::map<int, cv::Scalar> &getColorMap() {
  // OpenCV 使用 BGR 顺序
  static const std::map<int, cv::Scalar> kColorMap = {
      {0, cv::Scalar(0, 0, 0)},        // black
      {1, cv::Scalar(200, 0, 0)},      // background
      {2, cv::Scalar(102, 255, 100)},  // grass
      {3, cv::Scalar(0, 89, 118)},     // road
      {4, cv::Scalar(0, 255, 255)},    // dynamic
      {5, cv::Scalar(0, 0, 255)},      // static_obstacle
      {6, cv::Scalar(0, 165, 255)},    // wall obstacle
      {7, cv::Scalar(147, 20, 255)},   // vehicle obstacle
      {8, cv::Scalar(255, 255, 0)},    // pole
      {9, cv::Scalar(48, 130, 245)},   // impassable
      {10, cv::Scalar(128, 64, 0)},    // depression
      {11, cv::Scalar(34, 139, 34)},   // grass_bush
      {12, cv::Scalar(203, 192, 255)}, // limb_bush
      {13, cv::Scalar(226, 43, 138)},  // extra class (used in point cloud)

      {100, cv::Scalar(0, 0, 255)},   // pole        红色
      {101, cv::Scalar(0, 0, 255)},   // obst        红色
      {102, cv::Scalar(0, 0, 255)},   // fixo        红色
      {103, cv::Scalar(255, 0, 255)}, // car         洋红
      {104, cv::Scalar(0, 0, 255)},   // stat        红色
      {105, cv::Scalar(0, 255, 255)}, // dyna        黄色
      {106, cv::Scalar(255, 255, 0)}, // charge_station 青色
      {107, cv::Scalar(0, 255, 0)},   // person      绿色
  };

  return kColorMap;
}

} // namespace label_color

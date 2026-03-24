#include "offline_utils.hpp"

#include <cfloat>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

// ===================== PointCloudVisualizer =====================

void PointCloudVisualizer::saveThreeViews(
    const pcl::PointCloud<pcl::PointXYZRGBL> &cloud,
    const string &output_prefix)
{
  if (cloud.empty())
  {
    cout << "[Warning] Point cloud is empty, skip visualization." << endl;
    return;
  }

  float min_x = FLT_MAX, max_x = -FLT_MAX;
  float min_y = FLT_MAX, max_y = -FLT_MAX;
  float min_z = FLT_MAX, max_z = -FLT_MAX;

  for (const auto &pt : cloud.points)
  {
    if (std::isfinite(pt.x))
    {
      min_x = std::min(min_x, pt.x);
      max_x = std::max(max_x, pt.x);
    }
    if (std::isfinite(pt.y))
    {
      min_y = std::min(min_y, pt.y);
      max_y = std::max(max_y, pt.y);
    }
    if (std::isfinite(pt.z))
    {
      min_z = std::min(min_z, pt.z);
      max_z = std::max(max_z, pt.z);
    }
  }

  const int img_size = 800;
  const int margin = 50;

  Mat view_xy = Mat::zeros(img_size, img_size, CV_8UC3);
  Mat view_xz = Mat::zeros(img_size, img_size, CV_8UC3);
  Mat view_yz = Mat::zeros(img_size, img_size, CV_8UC3);

  view_xy.setTo(Scalar(255, 255, 255));
  view_xz.setTo(Scalar(255, 255, 255));
  view_yz.setTo(Scalar(255, 255, 255));

  float scale_x = (img_size - 2 * margin) / (max_x - min_x + 0.01f);
  float scale_y = (img_size - 2 * margin) / (max_y - min_y + 0.01f);
  float scale_z = (img_size - 2 * margin) / (max_z - min_z + 0.01f);

  for (const auto &pt : cloud.points)
  {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;

    cv::Vec3b color(pt.b, pt.g, pt.r);

    int px_xy = margin + static_cast<int>((pt.x - min_x) * scale_x);
    int py_xy = img_size - margin - static_cast<int>((pt.y - min_y) * scale_y);
    if (px_xy >= 0 && px_xy < img_size && py_xy >= 0 && py_xy < img_size)
    {
      view_xy.at<cv::Vec3b>(py_xy, px_xy) = color;
    }

    int px_xz = margin + static_cast<int>((pt.x - min_x) * scale_x);
    int pz_xz = img_size - margin - static_cast<int>((pt.z - min_z) * scale_z);
    if (px_xz >= 0 && px_xz < img_size && pz_xz >= 0 && pz_xz < img_size)
    {
      view_xz.at<cv::Vec3b>(pz_xz, px_xz) = color;
    }

    int py_yz = margin + static_cast<int>((pt.y - min_y) * scale_y);
    int pz_yz = img_size - margin - static_cast<int>((pt.z - min_z) * scale_z);
    if (py_yz >= 0 && py_yz < img_size && pz_yz >= 0 && pz_yz < img_size)
    {
      view_yz.at<cv::Vec3b>(pz_yz, py_yz) = color;
    }
  }

  putText(view_xy, "Top View (XY)", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
          Scalar(0, 0, 0), 2);
  putText(view_xz, "Side View (XZ)", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
          Scalar(0, 0, 0), 2);
  putText(view_yz, "Front View (YZ)", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
          Scalar(0, 0, 0), 2);

  imwrite(output_prefix + "_view_xy.jpg", view_xy);
  imwrite(output_prefix + "_view_xz.jpg", view_xz);
  imwrite(output_prefix + "_view_yz.jpg", view_yz);

  Mat combined(img_size, img_size * 3, CV_8UC3);
  view_xy.copyTo(combined(Rect(0, 0, img_size, img_size)));
  view_xz.copyTo(combined(Rect(img_size, 0, img_size, img_size)));
  view_yz.copyTo(combined(Rect(img_size * 2, 0, img_size, img_size)));
  imwrite(output_prefix + "_view_combined.jpg", combined);

  cout << "[Success] Saved 3-view images: " << output_prefix << "_view_*.jpg"
       << endl;
}

void PointCloudVisualizer::saveLabelDistribution(
    const pcl::PointCloud<pcl::PointXYZRGBL> &cloud)
{
  map<uint32_t, int> label_count;
  for (const auto &pt : cloud.points)
  {
    label_count[pt.label]++;
  }

  cout << "\n=== Point Cloud Label Distribution ===" << endl;
  cout << "Total points: " << cloud.size() << endl;
  for (const auto &pair : label_count)
  {
    double percentage = 100.0 * pair.second / cloud.size();
    cout << "Label " << pair.first << ": " << pair.second << " (" << fixed
         << setprecision(2) << percentage << "%)" << endl;
  }
}

// ===================== 标签分布工具 =====================

void printLabelDistribution(const cv::Mat &img_label)
{
  if (img_label.empty())
  {
    std::cout << "[Warning] Input label image is empty!" << std::endl;
    return;
  }

  if (img_label.channels() != 1)
  {
    std::cout << "[Warning] Expected single channel image, got "
              << img_label.channels() << " channels!" << std::endl;
    return;
  }

  std::map<int, int> label_count;

  for (int y = 0; y < img_label.rows; y++)
  {
    for (int x = 0; x < img_label.cols; x++)
    {
      int label_value = static_cast<int>(img_label.at<uchar>(y, x));
      label_count[label_value]++;
    }
  }

  int total_pixels = img_label.rows * img_label.cols;

  std::cout << "\n=== Label Distribution Analysis ===" << std::endl;
  std::cout << "Total pixels: " << total_pixels << std::endl;
  std::cout << "Number of unique labels: " << label_count.size() << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (const auto &pair : label_count)
  {
    int label = pair.first;
    int count = pair.second;
    double percentage = 100.0 * count / total_pixels;

    std::cout << "Label " << std::setw(3) << label << ": " << std::setw(8)
              << count << " pixels (" << std::fixed << std::setprecision(2)
              << std::setw(5) << percentage << "%)" << std::endl;
  }
  std::cout << "========================================" << std::endl;
}

std::map<int, int> getLabelDistribution(const cv::Mat &img_label)
{
  std::map<int, int> label_count;

  if (img_label.empty() || img_label.channels() != 1)
  {
    return label_count;
  }

  for (int y = 0; y < img_label.rows; y++)
  {
    for (int x = 0; x < img_label.cols; x++)
    {
      int label_value = static_cast<int>(img_label.at<uchar>(y, x));
      label_count[label_value]++;
    }
  }

  return label_count;
}

// ===================== 标签 + 检测融合 & 后处理 =====================

void filterLabelDect(Mat &src_lab, std::vector<Detection> &dect_src,
                     Mat &lab_dst, std::vector<Detection> &dect_dst,
                     bool enable_det)
{
  src_lab.copyTo(lab_dst);

  cv::Mat mask_zero;
  cv::compare(lab_dst, 0, mask_zero, cv::CMP_EQ);
  lab_dst.setTo(2, mask_zero);

  int shift_high = 370;
  cv::Rect force_region(0, shift_high, 640, 384 - shift_high);
  lab_dst(force_region).setTo(cv::Scalar(2));

  for (size_t i = 0; i < dect_src.size(); i++)
  {
    Bbox obj_box = dect_src[i].bbox;
    int8_t target_id = dect_src[i].id;

    int xmin = std::max(static_cast<int>(std::lround(obj_box.xmin)), 0);
    int ymin = std::max(static_cast<int>(std::lround(obj_box.ymin)), 0);
    int xmax =
        std::min(static_cast<int>(std::lround(obj_box.xmax)), lab_dst.cols - 1);
    int ymax =
        std::min(static_cast<int>(std::lround(obj_box.ymax)), lab_dst.rows - 1);

    if (xmin >= xmax || ymin >= ymax)
      continue;

    cv::Rect rect_tmp(xmin, ymin, xmax - xmin, ymax - ymin);
    cv::Mat roi_dst = lab_dst(rect_tmp);

    auto process_roi = [&](int set_value)
    {
      cv::Mat mask_one, mask_five;
      cv::inRange(roi_dst, cv::Scalar(1), cv::Scalar(1), mask_one);
      cv::inRange(roi_dst, cv::Scalar(5), cv::Scalar(5), mask_five);
      cv::Mat combined_mask = mask_one | mask_five;
      roi_dst.setTo(set_value, combined_mask);
    };

    if (target_id == 3)
    {
      process_roi(103);
    }
    else if (target_id == 6)
    {
      process_roi(106);
    }
    else if (target_id == 7)
    {
      // process_roi(107);
      cout << "[person] have been dect....." << endl;
    }
    else if (target_id == 4)  // 障碍物检测框 (id=4 -> label=104)
    {
      // 统计检测框内的标签分布
      int count_background = 0;     // label==1
      int count_walkable = 0;        // label==2 或 label==3 (草坪或道路)
      int total_pixels = 0;

      for (int y = 0; y < roi_dst.rows; y++)
      {
        for (int x = 0; x < roi_dst.cols; x++)
        {
          uint8_t label = roi_dst.at<uchar>(y, x);
          total_pixels++;
          if (label == 1) count_background++;
          else if (label == 2 || label == 3) count_walkable++;
        }
      }

      // 情况1: 检测框跨越背景(label==1)和可行走区域(label==2或3) - 过滤掉此检测框
      if (count_background > 0 && count_walkable > 0)
      {
        // 不添加到 dect_dst，直接跳过此检测框
        // 不对 label_map 做任何修改，保持原有分割结果
        continue;
      }
      // 情况2: 检测框完全在可行走区域(label==2或3)内
      else if (count_background == 0 && count_walkable > 0)
      {
        // 保留检测框标签，覆盖整个区域
        roi_dst.setTo(104);
        dect_dst.push_back(dect_src[i]);
      }
      // 情况3: 其他情况（主要是背景区域）
      else
      {
        // 只覆盖背景(1)和静态障碍物(5)
        cv::Mat mask_one, mask_five;
        cv::inRange(roi_dst, cv::Scalar(1), cv::Scalar(1), mask_one);
        cv::inRange(roi_dst, cv::Scalar(5), cv::Scalar(5), mask_five);
        cv::Mat combined_mask = mask_one | mask_five;
        roi_dst.setTo(104, combined_mask);
        dect_dst.push_back(dect_src[i]);
      }
    }
    else
    {
      if (enable_det)
      {
        roi_dst.setTo(target_id + 100);
        dect_dst.push_back(dect_src[i]);
      }
    }
  }
}

cv::Mat processLabelsOptimizedPipeline(const cv::Mat &label_img,
                                       const cv::Mat &rgb_img,
                                       float threshold)
{
  if (label_img.size() != rgb_img.size() || label_img.type() != CV_8UC1 ||
      rgb_img.type() != CV_8UC3)
  {
    std::cerr << "Error: Invalid input images!" << std::endl;
    return label_img.clone();
  }

  cv::Mat mask_label1, mask_label2, mask_label5;
  cv::compare(label_img, 1, mask_label1, cv::CMP_EQ);
  cv::compare(label_img, 2, mask_label2, cv::CMP_EQ);
  cv::compare(label_img, 5, mask_label5, cv::CMP_EQ);

  cv::Mat result_img = label_img.clone();

  cv::Mat hsv_img;
  cv::cvtColor(rgb_img, hsv_img, cv::COLOR_BGR2HSV);

  cv::Mat green_mask;
  cv::inRange(hsv_img, cv::Scalar(20, 40, 40), cv::Scalar(85, 255, 255),
              green_mask);

  // label == 5
  if (cv::countNonZero(mask_label5) > 0)
  {
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask_label5, mask_label5, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask_label5, mask_label5, cv::MORPH_OPEN, kernel);

    cv::Mat labels_cc, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(
        mask_label5, labels_cc, stats, centroids, 8, CV_32S);

    for (int i = 1; i < num_components; ++i)
    {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < 20)
        continue;

      int x = stats.at<int>(i, cv::CC_STAT_LEFT);
      int y = stats.at<int>(i, cv::CC_STAT_TOP);
      int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
      int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

      cv::Rect roi_rect(x, y, width, height);
      cv::Mat component_mask = (labels_cc(roi_rect) == i);
      cv::Mat green_mask_roi = green_mask(roi_rect);

      cv::Mat green_in_component;
      cv::bitwise_and(green_mask_roi, component_mask, green_in_component);

      int green_pixel_count = cv::countNonZero(green_in_component);
      float green_ratio = static_cast<float>(green_pixel_count) / area;

      if (green_ratio >= threshold)
      {
        cv::Mat final_mask;
        cv::bitwise_and(mask_label5, (labels_cc == i), final_mask);
        result_img.setTo(11, final_mask);
      }
    }
  }

  // label == 1
  if (cv::countNonZero(mask_label1) > 50)
  {
    cv::Mat label1_pixels;
    for (int y = 0; y < rgb_img.rows; ++y)
    {
      for (int x = 0; x < rgb_img.cols; ++x)
      {
        if (mask_label1.at<uchar>(y, x))
        {
          label1_pixels.push_back(rgb_img.at<cv::Vec3b>(y, x));
        }
      }
    }

    if (label1_pixels.rows > 10)
    {
      cv::Mat labels, centers;
      label1_pixels.convertTo(label1_pixels, CV_32F);
      cv::kmeans(
          label1_pixels, 2, labels,
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                           10, 1.0),
          3, cv::KMEANS_PP_CENTERS, centers);

      cv::Mat hsv_centers;
      centers.convertTo(centers, CV_8U);
      cv::cvtColor(centers.reshape(3, 2), hsv_centers, cv::COLOR_BGR2HSV);

      std::vector<bool> is_green_cluster(2, false);
      for (int i = 0; i < 2; ++i)
      {
        cv::Vec3b hsv_color = hsv_centers.at<cv::Vec3b>(i, 0);
        if (hsv_color[0] >= 30 && hsv_color[0] <= 85)
        {
          is_green_cluster[i] = true;
        }
      }

      cv::Mat clustered_mask = cv::Mat::zeros(label_img.size(), CV_8UC1);
      int pixel_idx = 0;

      for (int y = 0; y < rgb_img.rows; ++y)
      {
        for (int x = 0; x < rgb_img.cols; ++x)
        {
          if (mask_label1.at<uchar>(y, x))
          {
            int cluster_idx = labels.at<int>(pixel_idx++, 0);
            if (is_green_cluster[cluster_idx])
            {
              clustered_mask.at<uchar>(y, x) = 255;
            }
          }
        }
      }

      cv::Mat kernel =
          cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
      cv::morphologyEx(clustered_mask, clustered_mask, cv::MORPH_CLOSE, kernel);
      cv::morphologyEx(clustered_mask, clustered_mask, cv::MORPH_OPEN, kernel);

      cv::Mat labels_cc, stats, centroids;
      int num_components = cv::connectedComponentsWithStats(
          clustered_mask, labels_cc, stats, centroids, 8, CV_32S);

      for (int i = 1; i < num_components; ++i)
      {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > 50)
        {
          cv::Mat region_mask = (labels_cc == i);
          cv::bitwise_and(region_mask, mask_label1, region_mask);

          if (cv::countNonZero(region_mask) > 0)
          {
            result_img.setTo(11, region_mask);
          }
        }
      }
    }
  }

  // label == 2
  if (cv::countNonZero(mask_label2) > 0)
  {
    cv::Mat kernel_small =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask_label2, mask_label2, cv::MORPH_OPEN, kernel_small);

    cv::Mat kernel_large =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask_label2, mask_label2, cv::MORPH_CLOSE, kernel_large);

    result_img.setTo(2, mask_label2);
  }

  // label == 11 孔洞填充
  cv::Mat mask_label11;
  cv::compare(result_img, 11, mask_label11, cv::CMP_EQ);

  if (cv::countNonZero(mask_label11) > 0)
  {
    cv::Mat mask_filled = mask_label11.clone();
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask_filled, mask_filled, cv::MORPH_CLOSE, kernel);

    cv::Mat mask_inv;
    cv::bitwise_not(mask_filled, mask_inv);

    cv::Mat temp =
        cv::Mat::zeros(mask_filled.rows + 2, mask_filled.cols + 2, CV_8UC1);
    mask_inv.copyTo(temp(cv::Rect(1, 1, mask_filled.cols, mask_filled.rows)));
    cv::floodFill(temp, cv::Point(0, 0), cv::Scalar(255));
    cv::Mat holes = temp(cv::Rect(1, 1, mask_filled.cols, mask_filled.rows));
    cv::bitwise_not(holes, holes);

    cv::Mat final_mask;
    cv::bitwise_or(mask_filled, holes, final_mask);
    result_img.setTo(11, final_mask);
  }

  return result_img;
}

// ===================== CDT 矩形 =====================

cv::Rect get_cdt_rect(const std::vector<Detection> &ct_dect_src)
{
  if (ct_dect_src.empty() || ct_dect_src.size() >= 2)
  {
    return cv::Rect();
  }

  if (ct_dect_src.size() == 1)
  {
    int rect_xmin = ct_dect_src[0].bbox.xmin;
    int rect_xmax = ct_dect_src[0].bbox.xmax;
    int rect_ymin = ct_dect_src[0].bbox.ymin;
    int rect_ymax = ct_dect_src[0].bbox.ymax;

    int rect_width = rect_xmax - rect_xmin;
    int rect_heigh = 384 - rect_ymin;

    cv::Rect rect = cv::Rect(rect_xmin, rect_ymin, rect_width, rect_heigh);
    if (rect_xmin > 0 && rect_ymin > 0 && rect_xmax <= 640 &&
        rect_ymax <= 480)
    {
      return rect;
    }
    else
    {
      return cv::Rect();
    }
  }
  return cv::Rect();
}

// ===================== 语义可视化 & PCD 保存 =====================

void convertIdToRGB(const cv::Mat &img_lab, cv::Mat &parsing_img)
{
  // BGR 顺序，与 stereo_point_cloud::initColorMap() 保持一致
  static const std::map<int, cv::Scalar> cmap = {
      {0, cv::Scalar(0, 0, 0)},        // black
      {1, cv::Scalar(200, 0, 0)},      // background  蓝色
      {2, cv::Scalar(102, 255, 100)},  // grass       绿色
      {3, cv::Scalar(0, 89, 118)},     // road        褐色
      {4, cv::Scalar(0, 255, 255)},    // dynamic     黄色
      {5, cv::Scalar(0, 0, 255)},      // static_obstacle 红色
      {6, cv::Scalar(0, 165, 255)},    // wall obstacle
      {7, cv::Scalar(147, 20, 255)},   // vehicle obstacle
      {8, cv::Scalar(255, 255, 0)},    // pole
      {9, cv::Scalar(48, 130, 245)},   // impassable
      {10, cv::Scalar(128, 64, 0)},    // depression
      {11, cv::Scalar(34, 139, 34)},   // bush
      {12, cv::Scalar(203, 192, 255)}, // limb_bush
      {13, cv::Scalar(226, 43, 138)},  // CES_arod

      {100, cv::Scalar(0, 0, 255)},   // pole    红色
      {101, cv::Scalar(0, 0, 255)},   // obst    红色
      {102, cv::Scalar(0, 0, 255)},   // fixo    红色
      {103, cv::Scalar(255, 0, 255)}, // car     洋红
      {104, cv::Scalar(0, 0, 255)},   // stat    红色
      {105, cv::Scalar(0, 255, 255)}, // dyna    黄色
      {106, cv::Scalar(255, 255, 0)}, // charge_station 青色
  };

  for (int i = 0; i < img_lab.rows; ++i)
  {
    for (int j = 0; j < img_lab.cols; ++j)
    {
      int id = img_lab.at<uchar>(i, j);

      auto it = cmap.find(id);
      if (it != cmap.end())
      {
        const cv::Scalar &c = it->second; // BGR
        parsing_img.at<cv::Vec3b>(i, j) =
            cv::Vec3b(static_cast<uchar>(c[0]), static_cast<uchar>(c[1]),
                      static_cast<uchar>(c[2]));
      }
      else
      {
        parsing_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
      }
    }
  }
}

cv::Mat drawResult(Mat &img_src, Mat &img_lab, std::vector<Detection> &dect_src,
                   Mat &img_seg_show)
{
  static const std::map<int, std::string> mul_map_class = {
      {0, "unla"}, {1, "back"}, {2, "gras"}, {3, "road"}, {4, "dyna"}, {5, "stat"}, {6, "wall"}, {7, "vehi"}, {8, "pole"}, {9, "impa"}, {10, "depr"}, {11, "bush"}, {12, "limb"}, {13, "CES_arod"}, {14, "CES_stck"}, {15, "CES_pits"}, {100, "pole"}, {101, "obst"}, {102, "fixo"}, {103, "car"}, {104, "stat"}, {105, "dyna"}, {106, "chst"}, {107, "pers"}};

  cv::Mat parsing_img = Mat::zeros(img_lab.rows, img_lab.cols, CV_8UC3);
  Mat_<uint8_t> lawn_label = img_lab;
  (void)lawn_label; // currently unused but kept for compatibility

  convertIdToRGB(img_lab, parsing_img);
  for (size_t i = 0; i < dect_src.size(); i++)
  {
    int dect_num = dect_src[i].id;
    int mapped_id = dect_num + 100;
    std::string obj_name;
    auto it = mul_map_class.find(mapped_id);
    if (it != mul_map_class.end())
    {
      obj_name = it->second;
    }
    else
    {
      obj_name = "id_" + std::to_string(dect_num);
    }
    stringstream text_ss;
    text_ss << obj_name << ":" << std::fixed << std::setprecision(2)
            << dect_src[i].score;

    Bbox obj_box = dect_src.at(i).bbox;
    cv::Rect rect_tmp(obj_box.xmin, obj_box.ymin, (obj_box.xmax - obj_box.xmin),
                      (obj_box.ymax - obj_box.ymin));
    cv::rectangle(img_seg_show, rect_tmp, cv::Scalar(0, 255, 0), 2);
    cv::putText(
        img_seg_show, text_ss.str(), cv::Point(obj_box.xmin, obj_box.ymin + 15),
        FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  }
  cv::resize(parsing_img, parsing_img, img_src.size(), 0, 0, cv::INTER_NEAREST);

  float alpha_f = 0.6;
  addWeighted(img_src, alpha_f, parsing_img, 1 - alpha_f, 0.0, img_seg_show);
  return parsing_img;
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
//
// void convertIdToRGBOptimized(const cv::Mat &img_lab, cv::Mat &parsing_img) {
//     static const auto lut = getColorLookupTable();
//
//     int rows = img_lab.rows;
//     int cols = img_lab.cols;
//
//     for (int i = 0; i < rows; ++i) {
//         const uchar* row_ptr = img_lab.ptr<uchar>(i);
//         cv::Vec3b* out_ptr = parsing_img.ptr<cv::Vec3b>(i);
//         for (int j = 0; j < cols; ++j) {
//             out_ptr[j] = lut[row_ptr[j]];
//         }
//     }
// }
//
// cv::Mat drawResultOptimized(cv::Mat &img_src, cv::Mat &img_lab,
//                            std::vector<Detection> &dect_src, cv::Mat &img_seg_show) {
//
//     // 1. 生成颜色图 (在较小的尺寸上操作)
//     cv::Mat parsing_img(img_lab.size(), CV_8UC3);
//     convertIdToRGBOptimized(img_lab, parsing_img);
//
//     // 2. 将颜色图缩放到原图大小
//     // 如果 img_lab 和 img_src 尺寸一致，此步会自动跳过或非常快
//     if (parsing_img.size() != img_src.size()) {
//         cv::resize(parsing_img, parsing_img, img_src.size(), 0, 0, cv::INTER_NEAREST);
//     }
//
//     // 3. 图像融合 (Alpha Blending)
//     // 建议先融合，这样绘制的框和文字才不会被半透明遮盖
//     float alpha_f = 0.6f;
//     cv::addWeighted(img_src, alpha_f, parsing_img, 1.0f - alpha_f, 0.0, img_seg_show);
//
//     // 4. 在融合后的图上绘制检测框
//     for (const auto& det : dect_src) {
//         cv::Rect rect_tmp(det.bbox.xmin, det.bbox.ymin,
//                          (det.bbox.xmax - det.bbox.xmin),
//                          (det.bbox.ymax - det.bbox.ymin));
//
//         // 绘制矩形
//         cv::rectangle(img_seg_show, rect_tmp, cv::Scalar(0, 255, 0), 2);
//
//         // 优化文字拼接：避免使用 stringstream，简单场景用 to_string 更好
//         std::string label = "id_" + std::to_string(det.id) + ":" +
//                             cv::format("%.2f", det.score);
//
//         cv::putText(img_seg_show, label,
//                     cv::Point(det.bbox.xmin, std::max(det.bbox.ymin + 15, 15)),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
//     }
//
//     return parsing_img;
// }

Mat colorizeSegImg_test(Mat &img_src, Mat &img_lab, Mat &img_seg_show)
{
  cv::Mat parsing_img = Mat::zeros(img_lab.rows, img_lab.cols, CV_8UC3);
  uint8_t *parsing_img_ptr = parsing_img.ptr<uint8_t>();

  Mat_<uint8_t> lawn_label = img_lab;
  std::vector<cv::Mat> channels;
  cv::split(img_src, channels);
  for (int i = 0; i < img_lab.rows; i++)
  {
    for (int j = 0; j < img_lab.cols; j++)
    {
      int8_t id = lawn_label(i, j);
      *parsing_img_ptr++ = yj_seg_bgr_putpalette[id * 3];
      *parsing_img_ptr++ = yj_seg_bgr_putpalette[id * 3 + 1];
      *parsing_img_ptr++ = yj_seg_bgr_putpalette[id * 3 + 2];
    }
  }
  cv::resize(parsing_img, parsing_img, img_src.size(), 0, 0, cv::INTER_NEAREST);

  float alpha_f = 0.6;
  addWeighted(img_src, alpha_f, parsing_img, 1 - alpha_f, 0.0, img_seg_show);
  return parsing_img;
}

int savePcdfile_with_rgb_label(
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbi_cloud, string save_name)
{
  std::string pc_name = save_name + ".pcd";
  std::ofstream fout_pc_name(pc_name);

  fout_pc_name << "# .PCD v0.7 - Point Cloud Data file format" << std::endl;
  fout_pc_name << "VERSION 0.7" << std::endl;
  fout_pc_name << "FIELDS x y z rgb label" << std::endl;
  fout_pc_name << "SIZE 4 4 4 4 4" << std::endl;
  fout_pc_name << "TYPE F F F F U" << std::endl;
  fout_pc_name << "COUNT 1 1 1 1 1" << std::endl;
  fout_pc_name << "WIDTH " << xyz_rgbi_cloud.points.size() << std::endl;
  fout_pc_name << "HEIGHT 1" << std::endl;
  fout_pc_name << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
  fout_pc_name << "POINTS " << xyz_rgbi_cloud.points.size() << std::endl;
  fout_pc_name << "DATA ascii" << std::endl;

  for (auto &point : xyz_rgbi_cloud.points)
  {
    fout_pc_name << point.x << " " << point.y << " " << point.z << " "
                 << point.rgb << " " << point.label << std::endl;
  }
  fout_pc_name.close();
  return 1;
}

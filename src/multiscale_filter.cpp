#include "multiscale_filter.hpp"

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using Clock = std::chrono::high_resolution_clock;

// BFS connected component analysis
std::pair<std::vector<cv::Point>, cv::Mat>
bfsMarkConnected(const cv::Mat &disparity, int start_y, int start_x,
                 cv::Mat &visited, float max_diff = 5) {
  const int height = disparity.rows;
  const int width = disparity.cols;
  std::deque<cv::Point> points_queue;
  std::vector<cv::Point> connected_points;
  points_queue.push_back(cv::Point(start_x, start_y));

  const float center_value = disparity.at<float>(start_y, start_x);

  // 8-connectivity directions
  const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
  const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

  while (!points_queue.empty()) {
    cv::Point current = points_queue.front();
    points_queue.pop_front();

    if (visited.at<uchar>(current.y, current.x)) {
      continue;
    }

    visited.at<uchar>(current.y, current.x) = 1;
    connected_points.push_back(current);

    for (int dir = 0; dir < 8; ++dir) {
      int new_y = current.y + dy[dir];
      int new_x = current.x + dx[dir];

      if (new_y >= 0 && new_y < height && new_x >= 0 && new_x < width &&
          !visited.at<uchar>(new_y, new_x)) {

        float val = disparity.at<float>(new_y, new_x);
        if (!std::isnan(val) && std::abs(val - center_value) < max_diff) {
          points_queue.push_back(cv::Point(new_x, new_y));
        }
      }
    }
  }

  return {connected_points, visited};
}

cv::Mat smallRegionHighDisparityFilter(const cv::Mat &disparity,
                                       float disp_threshold = 50,
                                       int area_threshold = 200,
                                       cv::Mat obstacle_mask) {
  // Create binary mask for high disparity regions
  cv::Mat high_disp_mask = disparity > disp_threshold;

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(high_disp_mask, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Create output mask
  cv::Mat noise_filter_mask = cv::Mat::zeros(disparity.size(), CV_8U);

  // Fill small regions
  for (const auto &contour : contours) {
    // auto area = cv::contourArea(contour);
    // std::cout << "contour.size(): " << contour.size() << " , area: " << area
    // << std::endl;
    if (cv::contourArea(contour) < area_threshold) {
      // Check label mask
      if (!obstacle_mask.empty()) {
        cv::Mat contour_mask = cv::Mat::zeros(disparity.size(), CV_8U);
        cv::drawContours(contour_mask,
                         std::vector<std::vector<cv::Point>>{contour}, 0, 255,
                         cv::FILLED);
        int label_area = cv::countNonZero(contour_mask & obstacle_mask);
        if (label_area > 0.5 * cv::contourArea(contour)) {
          // Optional: remove non obstacle area
          // cv::Mat component_non_obstacle_mask = contour_mask &
          // ~obstacle_mask; noise_filter_mask |= component_non_obstacle_mask;
          continue;
        }
      }
      // std::cout << "remove this contour" << std::endl;

      cv::drawContours(noise_filter_mask,
                       std::vector<std::vector<cv::Point>>{contour}, 0, 255,
                       cv::FILLED);

      // cv::imshow("noise_filter_mask", noise_filter_mask);
      // cv::waitKey();
    }
  }

  // noise_filter_mask.convertTo(noise_filter_mask, CV_)

  return noise_filter_mask;
}

bool isLowTextureArea(const std::vector<cv::Point> &connected_points,
                      const cv::Mat &image) {
  // vector<Point> cv_points;
  // for (const auto& p : connected_points) {
  //     cv_points.push_back(Point(p.x, p.y));
  // }

  cv::Rect rect = cv::boundingRect(connected_points);
  cv::Mat block = image(rect);

  cv::Scalar mean, stddev;
  cv::meanStdDev(block, mean, stddev);
  double var = stddev[0] * stddev[0];

  return var < 20;
}

bool isLowTextureArea(const cv::Rect &roi, const cv::Mat &image) {

  // Calculate variance only for the roi region

  cv::Mat block = image(roi);

  cv::Scalar mean, stddev;
  cv::meanStdDev(block, mean, stddev);
  double std_val = stddev[0];
  // double var = stddev[0] * stddev[0];

  int maskedArea = cv::countNonZero(block);

  // std::cout << "maskedArea: " << maskedArea << " , mean: " << mean[0] << " ,
  // std: " << std_val << std::endl;

  return std_val < 25;
}

cv::Mat detectDisparityAnomaliesOptimized(const cv::Mat &reference_disparity,
                                          const cv::Mat &target_disparity,
                                          const MultiScaleFilterParams &params,
                                          const cv::Mat &gray,
                                          const cv::Mat obstacle_mask) {

  cv::Mat final_filter_mask = cv::Mat::zeros(target_disparity.size(), CV_8U);
  cv::Mat diff = target_disparity - reference_disparity;
  cv::Mat reference_disparity_valid = reference_disparity >= 0.0f;
  cv::Mat reference_disparity_invalid = reference_disparity < 0.0f;

  // Convert to binary mask for connected components
  cv::Mat potential_noise =
      (((diff > params.diff_threshold) & reference_disparity_valid) |
       ((diff > params.potential_threshold) & reference_disparity_invalid) |
       (target_disparity > params.excessive_threshold));
  cv::Mat noise_mask;
  potential_noise.convertTo(noise_mask, CV_8U);

  const int height = target_disparity.rows;
  const int width = target_disparity.cols;

  cv::Mat labels, stats, centroids;
  int num_components = connectedComponentsWithStats(noise_mask, labels, stats,
                                                    centroids, 4, CV_16U);

  for (int i = 1; i < num_components; i++) { // Start from 1 to skip background
    // Get component stats
    cv::Mat component_mask = (labels == i);
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    int left = stats.at<int>(i, cv::CC_STAT_LEFT);
    int top = stats.at<int>(i, cv::CC_STAT_TOP);
    int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    cv::Rect roi(left, top, width, height);

    // Visualize
    // cv::Mat component_disp = target_disparity.clone();
    // component_disp.setTo(0, ~component_mask);
    // cv::Mat obstacle_disp = target_disparity.clone();
    // obstacle_disp.setTo(0, ~component_mask);
    // cv::Mat obstacle_mask = obstacle_mask.clone();
    // obstacle_disp.setTo(0, ~obstacle_mask);
    // cv::Mat roi_disp = gray.clone();
    // // Set outside of roi to zero
    // roi_disp.setTo(0, ~component_mask);

    // cv::imshow("candidate_dilate", component_disp);
    // cv::imshow("roi", roi_disp);
    // cv::imshow("obstacle", obstacle_disp);
    // cv::waitKey();

    // Skip too large region/object
    if (area > 2000) {
      continue;
    }

    // Some small and far away candidates, we can safely remove them
    float mean_val = cv::mean(target_disparity, component_mask)[0];
    if ((area > 1000) && (mean_val < params.potential_threshold)) {
      final_filter_mask |= component_mask;
      continue;
    }

    // Skip processing if component is too large and not in low texture area
    if (area > 400) {
      // Create mask for current component
      if (!isLowTextureArea(roi, gray)) {
        continue;
      }

      // Check obstacle mask
      // If the component is likely an obstacle (area ratio > 0.5), skip it
      if (!obstacle_mask.empty()) {
        cv::Mat component_obstacle_mask = obstacle_mask & component_mask;
        int label_area = cv::countNonZero(component_obstacle_mask);
        // Only keep close obstacles
        if (label_area > 0.5 * area && mean_val > params.excessive_threshold) {
          // Optional: remove non obstacle area
          cv::Mat component_non_obstacle_mask = component_mask & ~obstacle_mask;
          final_filter_mask |= component_non_obstacle_mask;
          continue;
        }
      }
    }

    final_filter_mask |= component_mask;
  }

  return final_filter_mask;
}

// cv::Mat computeMultiScaleDisparity(cv::Ptr<cv::StereoBM>
// half_top_stereo_block_matcher,
//     cv::Ptr<cv::StereoSGBM> half_bottom_stereo_block_matcher,
//     const cv::Mat& gray_left, const cv::Mat& gray_right,
//     cv::Mat& input_disparity,
//     const MultiScaleFilterParams& orig_param, const MultiScaleFilterParams&
//     half_param, cv::Mat obstacle_mask) {

//     // Assume gray_left and gray_right are 320x240
//     int top_block_size = half_top_stereo_block_matcher->getBlockSize();
//     int bottom_block_size = half_bottom_stereo_block_matcher->getBlockSize();
//     cv::Mat disparity = input_disparity.clone();

//     auto t1 = Clock::now();

//     // Compute disparity
//     cv::Mat disparity_small_16S, disparity_small_32F;
//     cv::Mat disparity_half_top;
//     cv::Mat disparity_half_bottom;

//     // Process at 160x120
//     cv::Mat gray_left_half, gray_right_half;
//     cv::pyrDown(gray_left, gray_left_half);
//     cv::pyrDown(gray_right, gray_right_half);
//     int half_width = gray_left_half.cols; // 160
//     int half_height = gray_left_half.rows; // 120
//     disparity_small_32F.create(gray_left_half.size(), CV_32F);

//     cv::Rect half_top_with_pad(0, 0, half_width, half_height / 2 +
//     top_block_size); cv::Rect half_bottom_with_pad(0, half_height / 2 -
//     bottom_block_size, half_width, half_height / 2 + bottom_block_size);

//     cv::Mat gray_left_half_top = gray_left_half(half_top_with_pad);
//     cv::Mat gray_right_half_top = gray_right_half(half_top_with_pad);
//     cv::Mat gray_left_half_bottom = gray_left_half(half_bottom_with_pad);
//     cv::Mat gray_right_half_bottom = gray_right_half(half_bottom_with_pad);

//     half_top_stereo_block_matcher->compute(gray_left_half_top,
//     gray_right_half_top, disparity_half_top);
//     half_bottom_stereo_block_matcher->compute(gray_left_half_bottom,
//     gray_right_half_bottom, disparity_half_bottom);

//     disparity_half_top.convertTo(disparity_half_top, CV_32F);
//     disparity_half_top = disparity_half_top / 16.0f;
//     disparity_half_bottom.convertTo(disparity_half_bottom, CV_32F);
//     disparity_half_bottom = disparity_half_bottom / 16.0f;

//     cv::vconcat(disparity_half_top(cv::Rect(0, 0, half_width, half_height /
//     2)),
//         disparity_half_bottom(cv::Rect(0, bottom_block_size, half_width,
//         half_height / 2)), disparity_small_32F);

//     // disparity_small_32F(cv::Rect(0, 0, half_width, half_height / 2)) =
//     //     disparity_half_top(cv::Rect(0, 0, half_width, half_height / 2));
//     // disparity_small_32F(cv::Rect(0, half_height / 2, half_width,
//     half_height / 2)) =
//     //     disparity_half_bottom(cv::Rect(0, bottom_block_size, half_width,
//     half_height / 2));

//     // disparity_small_16S.convertTo(disparity_small_32F, CV_32F);
//     // disparity_small_32F /= 16.0f;
//     // std::cout << "disparity_small_32F.size: " <<
//     disparity_small_32F.size() << std::endl;
//     // Resize to 320x240 and do comparison
//     cv::resize(disparity_small_32F, disparity_small_32F,
//     input_disparity.size(), 0, 0, cv::INTER_NEAREST);
// //    cv::resize(disparity_small_32F, disparity_small_32F,
// input_disparity.size(), 0, 0, cv::INTER_LINEAR_EXACT);
//     disparity_small_32F *= 2;

//     // downsample obstacle_mask
//     cv::Mat obstacle_mask_small;
//     if(!obstacle_mask.empty()) {
//         cv::resize(obstacle_mask, obstacle_mask_small,
//         input_disparity.size(), 0, 0, cv::INTER_NEAREST);
//     }

//     auto t2 = Clock::now();
//     // cv::Mat orig_to_small_map =
//     detectDisparityAnomalies(disparity_small_32F, disparity.clone(),
//     orig_param, gray_left); cv::Mat orig_to_small_map =
//     detectDisparityAnomaliesOptimized(
//         disparity_small_32F, disparity.clone(), orig_param, gray_left,
//         obstacle_mask_small);
//     auto t3 = Clock::now();
//     // cv::Mat small_to_orig_map = detectDisparityAnomalies(disparity,
//     disparity_small_32F, half_param, gray_left); cv::Mat small_to_orig_map =
//     detectDisparityAnomaliesOptimized(
//         disparity, disparity_small_32F, half_param, gray_left,
//         obstacle_mask_small);
//     auto t4 = Clock::now();

// //    cv::imshow("orig_to_small_map", orig_to_small_map);
// //    cv::imshow("small_to_orig_map", small_to_orig_map);
// //    cv::waitKey();

//     // Apply filters
//     disparity.setTo(0, orig_to_small_map);
//     disparity.setTo(0, small_to_orig_map);

//     // Complete using small disparity
//     cv::Mat valid_for_complete = disparity < 0;
//     valid_for_complete = valid_for_complete & ~(orig_to_small_map |
//     small_to_orig_map); disparity_small_32F.copyTo(disparity,
//     valid_for_complete);

//     // Final noise filtering
//     // std::cout << "disparity size: " << disparity.size() << std::endl;
//     // double minVal, maxVal;
//     // cv::Point minLoc, maxLoc;

//     // cv::minMaxLoc(disparity, &minVal, &maxVal, &minLoc, &maxLoc);

//     // std::cout << "Minimum value: " << minVal << " at (" << minLoc.x << ",
//     " << minLoc.y << ")" << std::endl;
//     // std::cout << "Maximum value: " << maxVal << " at (" << maxLoc.x << ",
//     " << maxLoc.y << ")" << std::endl;

//     cv::Rect top = cv::Rect(0, 0, disparity.cols, half_height);
//     cv::Rect bottom = cv::Rect(0, half_height, disparity.cols, half_height);
//     cv::Mat noise_filter_mask0 = cv::Mat::zeros(disparity.size(), CV_8U);
//     cv::Mat noise_filter_mask1 = cv::Mat::zeros(disparity.size(), CV_8U);

//     if (!obstacle_mask.empty()) {
//         cv::Mat noise_filter_mask0_top =
//         smallRegionHighDisparityFilter(disparity(top), 12, 1000,
//         obstacle_mask_small(top)); cv::Mat noise_filter_mask0_bottom =
//         smallRegionHighDisparityFilter(disparity(bottom), 12, 150,
//         obstacle_mask_small(bottom));

//         cv::vconcat(noise_filter_mask0_top, noise_filter_mask0_bottom,
//         noise_filter_mask0);

//         cv::Mat noise_filter_mask1_top =
//         smallRegionHighDisparityFilter(disparity(top), 35, 250,
//         obstacle_mask_small(top)); cv::Mat noise_filter_mask1_bottom =
//         smallRegionHighDisparityFilter(disparity(bottom), 35, 250,
//         obstacle_mask_small(bottom));

//         cv::vconcat(noise_filter_mask1_top, noise_filter_mask1_bottom,
//         noise_filter_mask1);
//     }
//     else {
//         cv::Mat noise_filter_mask0_top =
//         smallRegionHighDisparityFilter(disparity(top), 12, 1000,
//         obstacle_mask_small); cv::Mat noise_filter_mask0_bottom =
//         smallRegionHighDisparityFilter(disparity(bottom), 12, 150,
//         obstacle_mask_small);

//         cv::vconcat(noise_filter_mask0_top, noise_filter_mask0_bottom,
//         noise_filter_mask0);

//         cv::Mat noise_filter_mask1_top =
//         smallRegionHighDisparityFilter(disparity(top), 35, 250,
//         obstacle_mask_small); cv::Mat noise_filter_mask1_bottom =
//         smallRegionHighDisparityFilter(disparity(bottom), 35, 250,
//         obstacle_mask_small);

//         cv::vconcat(noise_filter_mask1_top, noise_filter_mask1_bottom,
//         noise_filter_mask1);
//     }

//     // cv::imshow("noise_filter_mask0", noise_filter_mask0);
//     // cv::imshow("noise_filter_mask1", noise_filter_mask1);
//     // cv::waitKey();

//     // cv::Mat noise_filter_mask = smallRegionHighDisparityFilter(disparity,
//     12, 1000, obstacle_mask_small);
//     // cv::Mat noise_filter_mask2 = smallRegionHighDisparityFilter(disparity,
//     35, 250, obstacle_mask_small);
//     // noise_filter_mask = noise_filter_mask | noise_filter_mask2;
//     disparity.setTo(0, noise_filter_mask0);
//     disparity.setTo(0, noise_filter_mask1);
//     auto t5 = Clock::now();

//     // cv::imshow("noise_filter_mask", noise_filter_mask);
//     // cv::imshow("noise_filter_mask2", noise_filter_mask2);

//     // cv::Mat disparity_vis;
//     // cv::Mat disparity_colored;
//     // disparity.convertTo(disparity_colored, CV_8U, 255.0 / 100, 1);
//     // // cv::normalize(disparity_vis, disparity_colored, 0, 255,
//     cv::NORM_MINMAX, CV_8U);
//     // cv::applyColorMap(disparity_colored, disparity_colored,
//     cv::COLORMAP_VIRIDIS);

//     // cv::Mat disparity_half_vis;
//     // cv::Mat disparity_half_colored;
//     // disparity_small_32F.convertTo(disparity_half_colored, CV_8U, 255.0 /
//     100, 1);
//     // // cv::normalize(disparity_half_vis, disparity_half_colored, 0, 255,
//     cv::NORM_MINMAX, CV_8U);
//     // cv::applyColorMap(disparity_half_colored, disparity_half_colored,
//     cv::COLORMAP_VIRIDIS);

//     // cv::Mat disparity_orig_vis;
//     // cv::Mat disparity_orig_colored;
//     // input_disparity.convertTo(disparity_orig_colored, CV_8U, 255.0 / 100,
//     1);
//     // // cv::normalize(disparity_orig_vis, disparity_orig_colored, 0, 255,
//     cv::NORM_MINMAX, CV_8U);
//     // cv::applyColorMap(disparity_orig_colored, disparity_orig_colored,
//     cv::COLORMAP_VIRIDIS);

//     // cv::imshow("input_disparity", disparity_orig_colored);
//     // cv::imshow("disparity", disparity_colored);
//     // cv::imshow("disparity_small_32F", disparity_half_colored);
//     // cv::waitKey();

//     // Final resize
//     cv::resize(disparity, disparity, cv::Size(640, 480), 0, 0,
//     cv::INTER_NEAREST); disparity *= 2;

// //    std::cout << "small Disparity computation time: "
// //        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
// t1).count()
// //        << " ms" << std::endl;
// //    std::cout << "orig_to_small_map computation time: "
// //        << std::chrono::duration_cast<std::chrono::milliseconds>(t3 -
// t2).count()
// //        << " ms" << std::endl;
// //    std::cout << "small_to_orig_map computation time: "
// //        << std::chrono::duration_cast<std::chrono::milliseconds>(t4 -
// t3).count()
// //        << " ms" << std::endl;
// //    std::cout << "smallRegionHighDisparityFilter computation time: "
// //        << std::chrono::duration_cast<std::chrono::milliseconds>(t5 -
// t4).count()
// //        << " ms" << std::endl;

//     return disparity;
// }

cv::Mat computeMultiScaleDisparity(
    cv::Ptr<cv::StereoBM> half_top_stereo_block_matcher,
    cv::Ptr<cv::StereoSGBM> half_bottom_stereo_block_matcher,
    const cv::Mat &gray_left, const cv::Mat &gray_right,
    cv::Mat &input_disparity, const MultiScaleFilterParams &orig_param,
    const MultiScaleFilterParams &half_param, cv::Mat obstacle_mask, double fx,
    double fy, double cx, double cy, double baseline,
    bool enable_height_filter) {

  cv::Mat disparity = input_disparity.clone();

  // === 1. 计算小尺度视差 ===
  cv::Mat gray_left_half, gray_right_half;
  cv::pyrDown(gray_left, gray_left_half);
  cv::pyrDown(gray_right, gray_right_half);

  // === 2. 准备障碍物掩码 ===
  cv::Mat obstacle_mask_320;
  if (!obstacle_mask.empty() && obstacle_mask.size() != gray_left.size()) {
    cv::resize(obstacle_mask, obstacle_mask_320, gray_left.size(), 0, 0,
               cv::INTER_NEAREST);
  } else {
    obstacle_mask_320 = obstacle_mask;
  }

  int half_width = gray_left_half.cols;
  int half_height = gray_left_half.rows;

  // 分区域计算视差
  int top_block_size = half_top_stereo_block_matcher->getBlockSize();
  int bottom_block_size = half_bottom_stereo_block_matcher->getBlockSize();

  cv::Rect half_top_with_pad(0, 0, half_width,
                             half_height / 2 + top_block_size);
  cv::Rect half_bottom_with_pad(0, half_height / 2 - bottom_block_size,
                                half_width,
                                half_height / 2 + bottom_block_size);

  cv::Mat disparity_half_top, disparity_half_bottom;
  half_top_stereo_block_matcher->compute(gray_left_half(half_top_with_pad),
                                         gray_right_half(half_top_with_pad),
                                         disparity_half_top);
  half_bottom_stereo_block_matcher->compute(
      gray_left_half(half_bottom_with_pad),
      gray_right_half(half_bottom_with_pad), disparity_half_bottom);

  // 转换为32F并归一化
  disparity_half_top.convertTo(disparity_half_top, CV_32F, 1.0 / 16.0);
  disparity_half_bottom.convertTo(disparity_half_bottom, CV_32F, 1.0 / 16.0);

  // 组装完整的小尺度视差
  cv::Mat disparity_small_32F(half_height, half_width, CV_32F);
  disparity_half_top(cv::Rect(0, 0, half_width, half_height / 2))
      .copyTo(disparity_small_32F(cv::Rect(0, 0, half_width, half_height / 2)));
  disparity_half_bottom(
      cv::Rect(0, bottom_block_size, half_width, half_height / 2))
      .copyTo(disparity_small_32F(
          cv::Rect(0, half_height / 2, half_width, half_height / 2)));

  // === 3. 异常检测 (使用原始参数) ===
  cv::Mat input_disparity_small;
  cv::resize(input_disparity, input_disparity_small, gray_left_half.size(), 0,
             0, cv::INTER_NEAREST);
  input_disparity_small.convertTo(input_disparity_small, CV_32F,
                                  0.5f); // /= 2.0f

  cv::Mat obstacle_mask_small;
  if (!obstacle_mask_320.empty()) {
    cv::resize(obstacle_mask_320, obstacle_mask_small, gray_left_half.size(), 0,
               0, cv::INTER_NEAREST);
  }

  cv::Mat orig_to_small_map = detectDisparityAnomaliesOptimized(
      disparity_small_32F, input_disparity_small, orig_param, gray_left_half,
      obstacle_mask_small);

  cv::Mat small_to_orig_map = detectDisparityAnomaliesOptimized(
      input_disparity_small, disparity_small_32F, half_param, gray_left_half,
      obstacle_mask_small);

  // === 4. 应用异常检测结果 ===
  cv::resize(orig_to_small_map, orig_to_small_map, disparity.size(), 0, 0,
             cv::INTER_NEAREST);
  cv::resize(small_to_orig_map, small_to_orig_map, disparity.size(), 0, 0,
             cv::INTER_NEAREST);

  disparity.setTo(0, orig_to_small_map);
  disparity.setTo(0, small_to_orig_map);

  // === 5. 使用小尺度视差补全缺失区域 ===
  cv::Mat disparity_small_upscaled;
  cv::resize(disparity_small_32F, disparity_small_upscaled, disparity.size(), 0,
             0, cv::INTER_NEAREST);
  disparity_small_upscaled *= 2.0f;

  cv::Mat valid_for_complete =
      (disparity < 0) & ~(orig_to_small_map | small_to_orig_map);
  disparity_small_upscaled.copyTo(disparity, valid_for_complete);

  // === 6. 噪声过滤 (使用原始参数) ===
  cv::Rect top_roi(0, 0, disparity.cols, disparity.rows / 2);
  cv::Rect bottom_roi(0, disparity.rows / 2, disparity.cols,
                      disparity.rows / 2);

  cv::Mat noise_filter_mask = cv::Mat::zeros(disparity.size(), CV_8U);

  // 使用原始版本的过滤参数
  cv::Mat mask_top = smallRegionHighDisparityFilter(
      disparity(top_roi), 12, 1000,
      obstacle_mask_320.empty() ? cv::Mat() : obstacle_mask_320(top_roi));
  cv::Mat mask_bottom = smallRegionHighDisparityFilter(
      disparity(bottom_roi), 12, 150,
      obstacle_mask_320.empty() ? cv::Mat() : obstacle_mask_320(bottom_roi));

  mask_top.copyTo(noise_filter_mask(top_roi));
  mask_bottom.copyTo(noise_filter_mask(bottom_roi));

  // 可选: 添加第二级过滤 (如果需要)
  cv::Mat mask_top2 = smallRegionHighDisparityFilter(
      disparity(top_roi), 35, 250,
      obstacle_mask_320.empty() ? cv::Mat() : obstacle_mask_320(top_roi));
  cv::Mat mask_bottom2 = smallRegionHighDisparityFilter(
      disparity(bottom_roi), 35, 250,
      obstacle_mask_320.empty() ? cv::Mat() : obstacle_mask_320(bottom_roi));

  noise_filter_mask(top_roi) |= mask_top2;
  noise_filter_mask(bottom_roi) |= mask_bottom2;

  disparity.setTo(0, noise_filter_mask);

  // === 7. 高度约束过滤 (策略一) ===
  // 只有在enable_height_filter为true时才执行
  if (enable_height_filter) {
    // 目的：过滤掉地面区域被错误映射到空中的点
    cv::Mat height_filter_mask = cv::Mat::zeros(disparity.size(), CV_8U);

    // 可调参数
    const float ground_region_y_ratio = 0.5f; // 图像下半部分被认为是地面区域
    const float max_ground_height =
        -0.3f; // 地面点的Y坐标应该低于此值（相机坐标系，单位：米）
    const float sky_region_y_ratio = 0.3f;       // 图像上部分被认为是天空区域
    const float sky_high_disp_threshold = 30.0f; // 天空区域的高视差阈值

    int ground_start_row =
        static_cast<int>(disparity.rows * ground_region_y_ratio);
    int sky_end_row = static_cast<int>(disparity.rows * sky_region_y_ratio);

    // 内参缩放到当前视差图分辨率 (320x240)，原始内参对应 640x480
    double scale_x = static_cast<double>(disparity.cols) / 640.0;
    double scale_y = static_cast<double>(disparity.rows) / 480.0;
    double cx_scaled = cx * scale_x;
    double cy_scaled = cy * scale_y;
    double fx_scaled = fx * scale_x;
    double fy_scaled = fy * scale_y;

    for (int y = 0; y < disparity.rows; ++y) {
      for (int x = 0; x < disparity.cols; ++x) {
        float disp = disparity.at<float>(y, x);

        // 跳过无效视差
        if (disp <= 0 || std::isnan(disp)) {
          continue;
        }

        // 计算3D点的Y坐标（相机坐标系下的高度）
        // 注意：视差也是半分辨率，所以 baseline * fx_scaled / disp 等价于原始尺度
        float depth_Z = baseline * fx_scaled / disp;
        float point_Y = (y - cy_scaled) * depth_Z / fy_scaled;

        // 地面区域约束：如果像素在图像下半部分（可能是地面）
        // 但计算出的3D点高度却很高（Y坐标远大于相机），则标记为噪点
        if (y >= ground_start_row) {
          // 地面点的Y坐标应该接近或低于相机高度（负值表示在相机下方）
          // 如果Y坐标远高于相机（例如 > -0.3m），可能是错误映射
          if (point_Y > max_ground_height) {
            height_filter_mask.at<uchar>(y, x) = 255;
          }
        }

        // 天空区域约束：图像上部分出现非常近的点（高视差）通常是噪点
        if (y < sky_end_row && disp > sky_high_disp_threshold) {
          height_filter_mask.at<uchar>(y, x) = 255;
        }
      }
    }

    disparity.setTo(0, height_filter_mask);
  }

  // === 8. 最终上采样 ===
  cv::Mat disparity_final;
  cv::resize(disparity, disparity_final, cv::Size(640, 480), 0, 0,
             cv::INTER_NEAREST);
  disparity_final *= 2.0f;

  return disparity_final;
}

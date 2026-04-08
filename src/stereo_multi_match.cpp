#include "stereo_multi_match.h"
#include <opencv2/highgui.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

// #define VALID_HEIGHT 384
#define VALID_HEIGHT 432

void StereoMultiMatch::stereo_base_param_init()
{
    // Pl = (cv::Mat_<double>(3, 4) << 244.9567633, 0., 321.05016538, 0.,
    //       0.000000000000000, 244.9567633, 234.44410892, 0., 0.0, 0.0, 1.0, 0.0);
    // Pr = (cv::Mat_<double>(3, 4) << 244.9567633, 0., 321.05016538, -19.57480185,
    //       0.000000000000000, 244.9567633, 234.44410892, 0., 0.0, 0.0, 1.0, 0.0);

    // Pl = (cv::Mat_<double>(3, 4) << 344.4122655162, 0.0000000000, 343.8732070923, 0.0000000000,
    //       0.000000000000000, 303.8931754555, 238.2414806590, 0.0000000000, 0.0, 0.0, 1.0, 0.0);
    // Pr = (cv::Mat_<double>(3, 4) << 344.4122655162, 0.0000000000, 343.8732070923, -27.2907172705,
    //       0.000000000000000, 303.8931754555, 238.2414806590, 0.0000000000, 0.0, 0.0, 1.0, 0.0);

    Pl = (cv::Mat_<double>(3, 4) << 245.1634049359, 0.0000000000, 317.9649264254, 0.0000000000,
      0.0000000000, 245.1634049359, 242.8759116226, 0.0000000000, 0.0, 0.0, 1.0, 0.0);
    Pr = (cv::Mat_<double>(3, 4) << 245.1634049359, 0.0000000000, 317.9649264254, -19.5959393417,
      0.0000000000, 245.1634049359, 242.8759116226, 0.0000000000, 0.0, 0.0, 1.0, 0.0);

    cx = Pl.at<double>(0, 2); // Principal point x
    cy = Pl.at<double>(1, 2); // Principal point y
    fx = Pl.at<double>(0, 0); // Focal length x
    fy = Pl.at<double>(1, 1); // Focal length y
}

void StereoMultiMatch::stereo_dis_init()
{
    disparity_16S_.create(cv::Size(640, 480), CV_16S);
    disparity_.create(cv::Size(640, 480), CV_32F);
    depth_.create(cv::Size(640, 480), CV_32F);
    disparity_valid_mask_.create(cv::Size(640, 480), CV_8U);
    intensity_mask_.create(cv::Size(640, 480), CV_8U);
}

void StereoMultiMatch::stereo_block_matcher_init()
{
    stereo_block_matcher_ = cv::StereoBM::create();
    stereo_block_matcher_->setMinDisparity(0);
    stereo_block_matcher_->setNumDisparities(48);
    stereo_block_matcher_->setBlockSize(13);
    stereo_block_matcher_->setSpeckleWindowSize(64);
    stereo_block_matcher_->setSpeckleRange(16);
    stereo_block_matcher_->setDisp12MaxDiff(0);
    stereo_block_matcher_->setPreFilterType(1);
    stereo_block_matcher_->setPreFilterSize(15);
    stereo_block_matcher_->setPreFilterCap(31);
    stereo_block_matcher_->setTextureThreshold(5);
    stereo_block_matcher_->setUniquenessRatio(4);
}

void StereoMultiMatch::half_top_stereo_block_matcher_init()
{
    half_top_stereo_block_matcher_ = cv::StereoBM::create();
    half_top_stereo_block_matcher_->setMinDisparity(0);
    half_top_stereo_block_matcher_->setNumDisparities(32);
    half_top_stereo_block_matcher_->setBlockSize(13);
    half_top_stereo_block_matcher_->setSpeckleWindowSize(80);
    half_top_stereo_block_matcher_->setSpeckleRange(2);
    half_top_stereo_block_matcher_->setDisp12MaxDiff(0);
    half_top_stereo_block_matcher_->setPreFilterType(1);
    half_top_stereo_block_matcher_->setPreFilterSize(15);
    half_top_stereo_block_matcher_->setPreFilterCap(31);
    half_top_stereo_block_matcher_->setTextureThreshold(10);
    half_top_stereo_block_matcher_->setUniquenessRatio(10);
}

void StereoMultiMatch::half_bottom_stereo_block_matcher_init()
{
    half_bottom_stereo_block_matcher_ = cv::StereoSGBM::create();
    half_bottom_stereo_block_matcher_->setMinDisparity(0);
    half_bottom_stereo_block_matcher_->setNumDisparities(24);
    half_bottom_stereo_block_matcher_->setBlockSize(5);
    half_bottom_stereo_block_matcher_->setSpeckleWindowSize(64);
    half_bottom_stereo_block_matcher_->setSpeckleRange(16);
    half_bottom_stereo_block_matcher_->setDisp12MaxDiff(0);
    half_bottom_stereo_block_matcher_->setPreFilterCap(63);
    half_bottom_stereo_block_matcher_->setUniquenessRatio(20);
    half_bottom_stereo_block_matcher_->setP1(100);
    half_bottom_stereo_block_matcher_->setP2(500);
    half_bottom_stereo_block_matcher_->setMode(0);
}

void StereoMultiMatch::stereo_multi_param_init()
{
    stereo_dis_init();
    stereo_base_param_init();
    stereo_block_matcher_init();
    if (use_multiscale_filter_)
    {
        half_top_stereo_block_matcher_init();
        half_bottom_stereo_block_matcher_init();
    }

    orig_param_ = MultiScaleFilterParams(1.5, 4, 3, 1, 15);
    half_param_ = MultiScaleFilterParams(2.5, 8, 7, 3, 25);
}

// Image quality assessment for adaptive parameter selection
bool StereoMultiMatch::assess_image_quality(const cv::Mat& image, double& brightness, double& contrast)
{
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    brightness = mean[0];
    contrast = stddev[0];

    // Low quality: brightness < 50 or contrast < 20
    return (brightness >= 50.0 && contrast >= 20.0);
}

// StereoBM initialization with adaptive parameters (no hard distance cutoff)
void StereoMultiMatch::stereo_block_matcher_init_6m_adaptive(bool is_low_quality)
{
    stereo_block_matcher_ = cv::StereoBM::create();
    stereo_block_matcher_->setMinDisparity(0);  // Keep full range, filter by depth later
    stereo_block_matcher_->setNumDisparities(48);
    stereo_block_matcher_->setBlockSize(is_low_quality ? 17 : 13);

    // 增强 speckle 过滤以去除草坪噪点（温和水平）
    if (is_low_quality)
    {
        stereo_block_matcher_->setSpeckleWindowSize(100);  // 温和水平
        stereo_block_matcher_->setSpeckleRange(8);         // 温和水平
    }
    else
    {
        stereo_block_matcher_->setSpeckleWindowSize(64);
        stereo_block_matcher_->setSpeckleRange(16);
    }

    stereo_block_matcher_->setDisp12MaxDiff(0);
    stereo_block_matcher_->setPreFilterType(1);
    stereo_block_matcher_->setPreFilterSize(15);
    stereo_block_matcher_->setPreFilterCap(31);
    stereo_block_matcher_->setTextureThreshold(is_low_quality ? 10 : 5);  // 低质量图像提高纹理阈值
    stereo_block_matcher_->setUniquenessRatio(is_low_quality ? 10 : 4);
}

// Half-resolution top StereoBM initialization with adaptive parameters
void StereoMultiMatch::half_top_stereo_block_matcher_init_6m_adaptive(bool is_low_quality)
{
    half_top_stereo_block_matcher_ = cv::StereoBM::create();
    half_top_stereo_block_matcher_->setMinDisparity(0);  // Keep full range
    half_top_stereo_block_matcher_->setNumDisparities(32);
    half_top_stereo_block_matcher_->setBlockSize(is_low_quality ? 17 : 13);

    // 温和的 speckle 过滤
    if (is_low_quality)
    {
        half_top_stereo_block_matcher_->setSpeckleWindowSize(100);
        half_top_stereo_block_matcher_->setSpeckleRange(2);
    }
    else
    {
        half_top_stereo_block_matcher_->setSpeckleWindowSize(80);
        half_top_stereo_block_matcher_->setSpeckleRange(2);
    }

    half_top_stereo_block_matcher_->setDisp12MaxDiff(0);
    half_top_stereo_block_matcher_->setPreFilterType(1);
    half_top_stereo_block_matcher_->setPreFilterSize(15);
    half_top_stereo_block_matcher_->setPreFilterCap(31);
    half_top_stereo_block_matcher_->setTextureThreshold(10);
    half_top_stereo_block_matcher_->setUniquenessRatio(10);
}

// Half-resolution bottom StereoSGBM initialization with adaptive parameters
void StereoMultiMatch::half_bottom_stereo_block_matcher_init_6m_adaptive(bool is_low_quality)
{
    half_bottom_stereo_block_matcher_ = cv::StereoSGBM::create();
    half_bottom_stereo_block_matcher_->setMinDisparity(0);  // Keep full range
    half_bottom_stereo_block_matcher_->setNumDisparities(24);
    half_bottom_stereo_block_matcher_->setBlockSize(is_low_quality ? 9 : 5);

    // 温和的 speckle 过滤
    if (is_low_quality)
    {
        half_bottom_stereo_block_matcher_->setSpeckleWindowSize(80);
        half_bottom_stereo_block_matcher_->setSpeckleRange(10);
    }
    else
    {
        half_bottom_stereo_block_matcher_->setSpeckleWindowSize(64);
        half_bottom_stereo_block_matcher_->setSpeckleRange(16);
    }

    half_bottom_stereo_block_matcher_->setDisp12MaxDiff(0);
    half_bottom_stereo_block_matcher_->setPreFilterCap(63);
    half_bottom_stereo_block_matcher_->setUniquenessRatio(is_low_quality ? 30 : 20);
    half_bottom_stereo_block_matcher_->setP1(100);
    half_bottom_stereo_block_matcher_->setP2(500);
    half_bottom_stereo_block_matcher_->setMode(0);
}

// Parameter initialization with 6m limitation and adaptive quality assessment
void StereoMultiMatch::stereo_multi_param_init_6m_adaptive()
{
    stereo_dis_init();
    stereo_base_param_init();

    // Initialize with normal quality parameters (will be adjusted dynamically)
    stereo_block_matcher_init_6m_adaptive(false);

    if (use_multiscale_filter_)
    {
        half_top_stereo_block_matcher_init_6m_adaptive(false);
        half_bottom_stereo_block_matcher_init_6m_adaptive(false);
    }

    orig_param_ = MultiScaleFilterParams(1.5, 4, 3, 1, 15);
    half_param_ = MultiScaleFilterParams(2.5, 8, 7, 3, 25);
}

// Stereo processing with 6m limitation and adaptive parameters based on image quality
Mat StereoMultiMatch::stereo_multi_process_depth_6m_adaptive(Mat &rectifyL, Mat &rectifyR)
{
    // Assess image quality
    double brightness, contrast;
    bool is_good_quality = assess_image_quality(rectifyL, brightness, contrast);
    bool is_low_quality = !is_good_quality;

    // Re-initialize matchers with adaptive parameters based on quality
    stereo_block_matcher_init_6m_adaptive(is_low_quality);
    if (use_multiscale_filter_)
    {
        half_top_stereo_block_matcher_init_6m_adaptive(is_low_quality);
        half_bottom_stereo_block_matcher_init_6m_adaptive(is_low_quality);
    }

    temp_grayImageL = rectifyL.clone();

    // Image preprocessing
    if (bilateral_filter_kernel_size_ > 0)
    {
        cv::bilateralFilter(rectifyL, l_filtered_, bilateral_filter_kernel_size_,
                           bilateral_filter_sigma_color_, bilateral_filter_sigma_space_);
        cv::bilateralFilter(rectifyR, r_filtered_, bilateral_filter_kernel_size_,
                           bilateral_filter_sigma_color_, bilateral_filter_sigma_space_);
        rectifyL = l_filtered_;
        rectifyR = r_filtered_;
    }
    else if (use_background_substract_)
    {
        rectifyL = backgroundSubstract(rectifyL);
        rectifyR = backgroundSubstract(rectifyR);
    }

    // Downsample to half resolution
    cv::pyrDown(rectifyL, half_grayImageL);
    cv::pyrDown(rectifyR, half_grayImageR);

    // Initialize buffers based on current frame size
    if (disparity_16S_.empty() || disparity_16S_.size() != half_grayImageL.size())
    {
        disparity_16S_.create(half_grayImageL.size(), CV_16S);
    }

    // Compute disparity
    stereo_block_matcher_->compute(half_grayImageL, half_grayImageR, disparity_16S_);

    // Apply median filter if configured
    if (median_filter_kernel_size_ > 0)
    {
        cv::medianBlur(disparity_16S_, disparity_16S_, median_filter_kernel_size_);
    }

    // Convert disparity to CV_32F
    if (disparity_.empty() || disparity_.size() != disparity_16S_.size())
    {
        disparity_.create(disparity_16S_.size(), CV_32F);
    }
    disparity_16S_.convertTo(disparity_, CV_32F);
    disparity_ = (disparity_ / 16.0f);

    return disparity_;
}

cv::Mat StereoMultiMatch::backgroundSubstract(const cv::Mat &image)
{
    cv::Mat gray_image;
    cv::Mat preprocessed_image;

    // Convert image to grayscale if it's a color image
    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray_image = image.clone(); // If already grayscale, use a copy
    }

    // Create large Gaussian kernel
    int kernel_size = 15;
    double sigma_large = 3;
    cv::Mat large_gaussian_blurred;

    // Apply Gaussian blur
    cv::GaussianBlur(gray_image, large_gaussian_blurred,
                     cv::Size(kernel_size, kernel_size), sigma_large);

    // Background subtraction: Original image minus large Gaussian blurred image
    cv::Mat float_image, float_blurred, subtracted_image;
    gray_image.convertTo(float_image, CV_32F);
    large_gaussian_blurred.convertTo(float_blurred, CV_32F);

    cv::subtract(float_image, float_blurred, subtracted_image);

    // Normalize the result to full range [0, 255]
    cv::normalize(subtracted_image, preprocessed_image, 0, 255, cv::NORM_MINMAX);

    // Convert the image back to 8-bit
    preprocessed_image.convertTo(preprocessed_image, CV_8U);

    return preprocessed_image;
}

cv::Mat StereoMultiMatch::stereo_multi_process(cv::Mat &rectifyL,
                                               cv::Mat &rectifyR,
                                               bool enable_height_filter_)
{
    if (rectifyL.empty() || rectifyR.empty())
    {
        throw std::runtime_error("Input images are empty!");
    }
    if (rectifyL.size() != rectifyR.size())
    {
        throw std::runtime_error("Input images have different sizes!");
    }

    const cv::Size orig_size = rectifyL.size();

    cv::Mat rectifyL_orig = rectifyL.clone();

    // 预处理图像
    cv::Mat rectifyL_processed, rectifyR_processed;

    // 根据配置选择不同预处理方式
    if (bilateral_filter_kernel_size_ > 0)
    {
        cv::bilateralFilter(
            rectifyL, rectifyL_processed, bilateral_filter_kernel_size_,
            bilateral_filter_sigma_color_, bilateral_filter_sigma_space_);
        cv::bilateralFilter(
            rectifyR, rectifyR_processed, bilateral_filter_kernel_size_,
            bilateral_filter_sigma_color_, bilateral_filter_sigma_space_);
    }
    else if (use_background_substract_)
    {
        rectifyL_processed = backgroundSubstract(rectifyL);
        rectifyR_processed = backgroundSubstract(rectifyR);
    }
    else
    {
        rectifyL.copyTo(rectifyL_processed);
        rectifyR.copyTo(rectifyR_processed);
    }

    // 降采样
    cv::pyrDown(rectifyL_processed, rectifyL_processed);
    cv::pyrDown(rectifyR_processed, rectifyR_processed);

    // 复用或调整缓冲区大小
    if (disparity_16S_.empty() ||
        disparity_16S_.size() != rectifyL_processed.size())
        disparity_16S_.create(rectifyL_processed.size(), CV_16S);

    // 计算视差
    stereo_block_matcher_->compute(rectifyL_processed, rectifyR_processed,
                                   disparity_16S_);

    // 中值滤波
    if (median_filter_kernel_size_ > 0)
    {
        cv::medianBlur(disparity_16S_, disparity_16S_, median_filter_kernel_size_);
    }

    // 转换视差数据
    if (disparity_.empty() || disparity_.size() != disparity_16S_.size())
        disparity_.create(disparity_16S_.size(), CV_32F);
    disparity_16S_.convertTo(disparity_, CV_32F, 1.0f / 16.0f);

    // 多尺度处理或调整尺寸
    if (use_multiscale_filter_)
    {
        // 计算baseline: baseline = |Pr(0,3)| / fx
        double baseline = std::abs(Pr.at<double>(0, 3)) / fx;

        disparity_ = computeMultiScaleDisparity(
            half_top_stereo_block_matcher_, half_bottom_stereo_block_matcher_,
            rectifyL_processed, rectifyR_processed, disparity_, orig_param_,
            half_param_, cv::Mat(), fx, fy, cx, cy, baseline,
            enable_height_filter_);
    }
    else if (disparity_.size() != orig_size)
    {
        cv::resize(disparity_, disparity_, orig_size, 0, 0, cv::INTER_NEAREST);
    }

    // 创建 mask 缓冲区
    if (disparity_valid_mask_.empty() ||
        disparity_valid_mask_.size() != disparity_.size())
        disparity_valid_mask_.create(disparity_.size(), CV_8U);
    disparity_valid_mask_ = (disparity_ > 0.01);

    if (intensity_mask_.empty() || intensity_mask_.size() != rectifyL_orig.size())
        intensity_mask_.create(rectifyL_orig.size(), CV_8U);
    intensity_mask_ =
        (rectifyL_orig >= intensity_low_) & (rectifyL_orig <= intensity_high_);

    // 合并有效区域
    disparity_valid_mask_ &= intensity_mask_;

    // 计算深度图
    static thread_local cv::Mat depth_; // 使用 thread_local 避免多线程冲突
    if (depth_.empty() || depth_.size() != disparity_.size())
    {
        depth_.release(); // 显式释放旧内容
        depth_.create(disparity_.size(), CV_32F);
    }

    double Bf = abs(Pr.at<double>(0, 3));          // baseline * focal length
    cv::divide(Bf, disparity_, depth_, 1, CV_32F); // 避免除零错误
    depth_.setTo(100.0f, ~disparity_valid_mask_);

    // 应用形态学操作去除孤立点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat opened_depth, closed_depth;

    // 开运算：先腐蚀后膨胀，去除小噪声点
    cv::morphologyEx(depth_, opened_depth, cv::MORPH_OPEN, kernel);
    // 闭运算：先膨胀后腐蚀，填补小孔洞
    cv::morphologyEx(opened_depth, closed_depth, cv::MORPH_CLOSE, kernel);
    depth_ = closed_depth;

    // 释放中间变量以减少内存占用
    rectifyL_processed.release();
    rectifyR_processed.release();

    return depth_.clone(); // 或者使用 std::move(depth_)
}

Mat StereoMultiMatch::stereo_multi_process_depth(Mat &rectifyL, Mat &rectifyR)
{
    temp_grayImageL = rectifyL.clone();

    if (bilateral_filter_kernel_size_ > 0)
    {
        cv::bilateralFilter(rectifyL, l_filtered_, bilateral_filter_kernel_size_,
                            bilateral_filter_sigma_color_,
                            bilateral_filter_sigma_space_);

        cv::bilateralFilter(rectifyR, r_filtered_, bilateral_filter_kernel_size_,
                            bilateral_filter_sigma_color_,
                            bilateral_filter_sigma_space_);

        rectifyL = l_filtered_;
        rectifyR = r_filtered_;
    }
    else if (use_background_substract_)
    {
        rectifyL = backgroundSubstract(rectifyL);
        rectifyR = backgroundSubstract(rectifyR);
    }

    // To 320x240 or 320x216 (adaptive)
    cv::pyrDown(rectifyL, half_grayImageL);
    cv::pyrDown(rectifyR, half_grayImageR);

    // Initialize buffers based on current frame size to avoid crash when
    // switching modes
    if (disparity_16S_.empty() ||
        disparity_16S_.size() != half_grayImageL.size())
    {
        disparity_16S_.create(half_grayImageL.size(), CV_16S);
    }

    // Calculate disparity
    stereo_block_matcher_->compute(half_grayImageL, half_grayImageR,
                                   disparity_16S_);

    if (median_filter_kernel_size_ > 0)
    {
        cv::medianBlur(disparity_16S_, disparity_16S_, median_filter_kernel_size_);
    }

    // Converting disparity values to CV_32F from CV_16S and scaling down
    if (disparity_.empty() || disparity_.size() != disparity_16S_.size())
    {
        disparity_.create(disparity_16S_.size(), CV_32F);
    }
    disparity_16S_.convertTo(disparity_, CV_32F);
    disparity_ = (disparity_ / 16.0f);
    return disparity_;
}

Mat StereoMultiMatch::stereo_multi_process_filter(Mat &disparity_,
                                                  cv::Mat lab_dst,
                                                  bool enable_height_filter_)
{
    // 让多尺度立体匹配在障碍物 ROI 上保守一些，避免近场障碍物被当成噪声滤掉。
    cv::Mat label_mask =
        (lab_dst == 4) | (lab_dst == 5) | (lab_dst >= 100);
    label_mask.convertTo(label_mask, CV_8UC1, 255);

    // label_mask 可能是 640x384（分割模型输出），需要 padding 到 640x480 与深度图对齐
    // 底部填充0（无障碍物），确保 resize 到 320x240 时空间坐标不错位
    cv::Size depth_full_size = temp_grayImageL.size(); // 640x480
    if (!label_mask.empty() && label_mask.rows < depth_full_size.height)
    {
        cv::Mat padded_label_mask = cv::Mat::zeros(depth_full_size, CV_8UC1);
        label_mask.copyTo(padded_label_mask(cv::Rect(0, 0, label_mask.cols, label_mask.rows)));
        label_mask = padded_label_mask;
    }

    if (use_multiscale_filter_)
    {
        // 计算baseline: baseline = |Pr(0,3)| / fx
        double baseline = std::abs(Pr.at<double>(0, 3)) / fx;

        disparity_ = computeMultiScaleDisparity(
            half_top_stereo_block_matcher_, half_bottom_stereo_block_matcher_,
            half_grayImageL, half_grayImageR, disparity_, orig_param_, half_param_,
            label_mask, fx, fy, cx, cy, baseline, enable_height_filter_);
    }
    else
    {
        cv::Size orig_size = temp_grayImageL.size(); // 动态尺寸
        cv::resize(disparity_, disparity_, orig_size, 0, 0, cv::INTER_NEAREST);
    }

    // Re-allocate buffers if the size has changed
    if (disparity_valid_mask_.empty() ||
        disparity_valid_mask_.size() != disparity_.size())
    {
        disparity_valid_mask_.create(disparity_.size(), CV_8U);
    }
    if (intensity_mask_.empty() ||
        intensity_mask_.size() != temp_grayImageL.size())
    {
        intensity_mask_.create(temp_grayImageL.size(), CV_8U);
    }
    if (depth_.empty() || depth_.size() != disparity_.size())
    {
        depth_.create(disparity_.size(), CV_32F);
    }

    disparity_valid_mask_ = (disparity_ > 0.01); // Valid disparities

    intensity_mask_ = (temp_grayImageL >= intensity_low_) &
                      (temp_grayImageL <= intensity_high_);
    disparity_valid_mask_ = disparity_valid_mask_ & intensity_mask_;

    double Bf = abs(Pr.at<double>(0, 3)); // baseline * focal length
    cv::divide(Bf, disparity_, depth_, CV_32F);
    depth_.setTo(100.0f, ~disparity_valid_mask_);

    // Max depth cutoff in visualization
    float MAX_DEPTH = 10.0;
    cv::Mat max_depth_mask = depth_ > MAX_DEPTH;
    depth_.setTo(MAX_DEPTH, max_depth_mask);
    return depth_;
}

bool StereoMultiMatch::in_range(const cv::Point &top_left,
                                const cv::Point &bottom_right,
                                const cv::Point &pt_2d)
{
    int u = pt_2d.x, v = pt_2d.y;
    if (u < top_left.x || v < top_left.y)
        return false;
    if (u > bottom_right.x || v > bottom_right.y)
        return false;
    return true;
}

void StereoMultiMatch::det_pc_rgb_label(Detection &det, const cv::Point &pt,
                                        pcl::PointXYZRGBL &pci)
{
    Point left(det.bbox.xmin, det.bbox.ymin);
    Point right(det.bbox.xmax, det.bbox.ymax);
    if (in_range(left, right, pt))
    {
        int temp_id = det.id + 100;
        if (temp_id < 200)
        {
            pci.label = temp_id;
        }
    }
}

void StereoMultiMatch::stereo_process_pc_rgbl_depth(
    const Mat &depth, Mat &ori_mat,
    pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
    pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud)
{
    //    for (int y = 0; y < depth_.rows; y+=2) {
    for (int y = 0; y < depth.rows; y += 3)
    {
        for (int x = 0; x < depth.cols; x += 3)
        {
            float d = depth.at<float>(y, x);
            pcl::PointXYZRGBL pc_rgbl;
            if (d >= 0 && d < 6.0f)
            {
                // Valid depth
                pc_rgbl.x = (x - cx) * d / fx;
                pc_rgbl.y = (y - cy) * d / fy;
                pc_rgbl.z = d;
                Vec3b color = ori_mat.at<Vec3b>(y, x);
                uint8_t r = static_cast<uint8_t>(color[2]); // 红色分量
                uint8_t g = static_cast<uint8_t>(color[1]); // 绿色分量
                uint8_t b = static_cast<uint8_t>(color[0]); // 蓝色分量

                uint32_t rgb_packed =
                    ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                pc_rgbl.rgb = *reinterpret_cast<float *>(&rgb_packed);
                //                pc_rgbl.label = (int) lab.at<uchar>(y, x);

                pc_rgbl.label = 2;
                if (pc_rgbl.label <= 0 || pc_rgbl.label > 200)
                    continue;
                xyz_rgbl_cloud.push_back(pc_rgbl);
            }
        }
    }

    stereo_point_ori_rgb_filter(xyz_rgbl_cloud, out_xyz_rgbl_cloud);

    out_xyz_rgbl_cloud.height = 1;
    out_xyz_rgbl_cloud.width = out_xyz_rgbl_cloud.points.size();
    out_xyz_rgbl_cloud.points.resize(out_xyz_rgbl_cloud.width *
                                     out_xyz_rgbl_cloud.height);
    //    cout << "point xyzi_cloud size = " << xyzi_cloud.points.size() << endl;
}

void StereoMultiMatch::stereo_process_pc_rgbl_dest(
    const Mat &depth, Mat &ori_mat, std::vector<Detection> &dect_src,
    pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
    pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud)
{
    for (int y = 0; y < depth.rows; y += 3)
    {
        for (int x = 0; x < depth.cols; x += 3)
        {
            float d = depth.at<float>(y, x);
            pcl::PointXYZRGBL pc_rgbl;
            if (d >= 0 && d < 6.0f)
            {
                // Valid depth
                pc_rgbl.x = (x - cx) * d / fx;
                pc_rgbl.y = (y - cy) * d / fy;
                pc_rgbl.z = d;

                Vec3b color = ori_mat.at<Vec3b>(y, x);
                uint8_t r = static_cast<uint8_t>(color[2]); // 红色分量
                uint8_t g = static_cast<uint8_t>(color[1]); // 绿色分量
                uint8_t b = static_cast<uint8_t>(color[0]); // 蓝色分量

                uint32_t rgb_packed =
                    ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                pc_rgbl.rgb = *reinterpret_cast<float *>(&rgb_packed);
                pc_rgbl.label = 2;
                if (pc_rgbl.label <= 0 || pc_rgbl.label > 200)
                    continue;
                for (int i = 0; i < dect_src.size(); i++)
                {
                    det_pc_rgb_label(dect_src[i], cv::Point(x, y), pc_rgbl);
                }
                xyz_rgbl_cloud.push_back(pc_rgbl);
            }
        }
    }

    stereo_point_ori_rgb_filter(xyz_rgbl_cloud, out_xyz_rgbl_cloud);

    out_xyz_rgbl_cloud.height = 1;
    out_xyz_rgbl_cloud.width = out_xyz_rgbl_cloud.points.size();
    out_xyz_rgbl_cloud.points.resize(out_xyz_rgbl_cloud.width *
                                     out_xyz_rgbl_cloud.height);
}

void StereoMultiMatch::stereo_process_pci_depth_rgb_seg_fusion(
    const Mat &depth, const Mat &lab, Mat &ori_mat,
    pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
    pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud)
{
    for (int y = 0; y < depth.rows; y += 4)
    {
        for (int x = 0; x < depth.cols; x += 4)
        {
            float d = depth.at<float>(y, x);
            pcl::PointXYZRGBL pc_rgbl;
            if (d > 0 && d < 6.0f)
            {
                // Valid depth
                pc_rgbl.x = (x - cx) * d / fx;
                pc_rgbl.y = (y - cy) * d / fy;
                pc_rgbl.z = d;

                Vec3b color = ori_mat.at<Vec3b>(y, x);
                uint8_t r = static_cast<uint8_t>(color[2]); // 红色分量
                uint8_t g = static_cast<uint8_t>(color[1]); // 绿色分量
                uint8_t b = static_cast<uint8_t>(color[0]); // 蓝色分量

                uint32_t rgb_packed =
                    ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                pc_rgbl.rgb = *reinterpret_cast<float *>(&rgb_packed);
                pc_rgbl.label = (int)lab.at<uchar>(y, x);
                if (pc_rgbl.label <= 0 || pc_rgbl.label > 200)
                    continue;
                xyz_rgbl_cloud.push_back(pc_rgbl);
            }
        }
    }

    stereo_point_ori_rgb_filter(xyz_rgbl_cloud, out_xyz_rgbl_cloud);

    out_xyz_rgbl_cloud.height = 1;
    out_xyz_rgbl_cloud.width = out_xyz_rgbl_cloud.points.size();
    out_xyz_rgbl_cloud.points.resize(out_xyz_rgbl_cloud.width *
                                     out_xyz_rgbl_cloud.height);
};

void StereoMultiMatch::stereo_process_pci_depth_rgb_seg_det_fusion(
    const Mat &depth, const Mat &lab, std::vector<Detection> &,
    Mat &ori_mat, pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
    pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud)
{
    int safe_rows = std::min({VALID_HEIGHT, depth.rows, lab.rows, ori_mat.rows});
    int safe_cols = std::min({depth.cols, lab.cols, ori_mat.cols});

    // 统计障碍物点云增强效果
    int obstacle_points_added = 0;
    int total_points_added = 0;

    // ========== 计算各障碍物label的近端深度参考（用于拖影过滤）==========
    // 用较小分位数而不是中值，优先保留近端前景，抑制把后方背景当主体深度。
    std::map<uint8_t, float> label_depth_reference;
    {
        std::map<uint8_t, std::vector<float>> label_depths;
        for (int y = 0; y < safe_rows; y++)
        {
            for (int x = 0; x < safe_cols; x++)
            {
                uint8_t label = lab.at<uchar>(y, x);
                if (label < 100) continue;
                float d = depth.at<float>(y, x);
                if (d > 0 && d < 6.0f)
                    label_depths[label].push_back(d);
            }
        }
        for (auto &[label, depths] : label_depths)
        {
            if (!depths.empty())
            {
                const size_t near_index =
                    std::min(depths.size() - 1, depths.size() / 4);
                std::nth_element(depths.begin(), depths.begin() + near_index,
                                 depths.end());
                label_depth_reference[label] = depths[near_index];
                std::cout << "[DepthFilter] Label " << static_cast<int>(label)
                          << " 近端深度参考(P25): "
                          << label_depth_reference[label] << "m" << std::endl;
            }
        }
    }
    const float far_depth_tolerance = 0.35f; // 只抑制后方拖尾，近端前景尽量保留

    for (int y = 0; y < safe_rows; y += 4)
    {
        for (int x = 0; x < safe_cols; x += 4)
        {
            float d = depth.at<float>(y, x);
            pcl::PointXYZRGBL pc_rgbl;
            if (d > 0 && d < 6.0f)
            {
                // Valid depth
                pc_rgbl.x = (x - cx) * d / fx;
                pc_rgbl.y = (y - cy) * d / fy;
                pc_rgbl.z = d;

                Vec3b color = ori_mat.at<Vec3b>(y, x);
                uint8_t r = static_cast<uint8_t>(color[2]); // 红色分量
                uint8_t g = static_cast<uint8_t>(color[1]); // 绿色分量
                uint8_t b = static_cast<uint8_t>(color[0]); // 蓝色分量

                uint32_t rgb_packed =
                    ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                pc_rgbl.rgb = *reinterpret_cast<float *>(&rgb_packed);
                pc_rgbl.label = lab.at<uchar>(y, x);

                // 底部边角草坪过滤：仅当label==2或3且位于底部边角时跳过，
                // 保留障碍物(其他label)信息，仅减少底部边角草坪点云，
                // 避免建图时机器人因边角草坪左右摇摆
                if ((pc_rgbl.label == 2 || pc_rgbl.label == 3) && y > safe_rows * 4 / 5)
                {
                    int edge_w = 48 + (y - safe_rows * 4 / 5) * 120 / (safe_rows * 2 / 5);
                    if (x < edge_w || x > safe_cols - edge_w)
                        continue;
                }
                if (pc_rgbl.label <= 0 || pc_rgbl.label > 200)
                    continue;

                // 策略1：仅针对 label==1（背景/草坪）的深度范围过滤
                // label==3（道路）是可通行区域，不能过滤
                if (pc_rgbl.label == 1)
                {
                    // 更激进：过滤所有近距离背景（< 3.5m）
                    if (d < 3.5f)
                    {
                        continue;
                    }
                }

                // 策略2：仅针对 label==1（背景/草坪）的高度过滤
                if (pc_rgbl.label == 1)
                {
                    // 过滤地面附近和高处的背景点
                    // Y > -0.3: 地面附近或高于地面（更严格）
                    // Y < -0.6: 过高的悬浮点（更严格）
                    if (pc_rgbl.y > -0.3f || pc_rgbl.y < -0.6f)
                    {
                        continue;
                    }
                }

                // 邻域一致性检查：仅针对 label==1（背景）
                if (pc_rgbl.label == 1)
                {
                    int valid_neighbors = 0;
                    for (int dy = -4; dy <= 4; dy += 4)
                    {
                        for (int dx = -4; dx <= 4; dx += 4)
                        {
                            if (dy == 0 && dx == 0)
                                continue;
                            int ny = y + dy, nx = x + dx;
                            if (ny >= 0 && ny < safe_rows && nx >= 0 && nx < safe_cols)
                            {
                                uint8_t neighbor_label = lab.at<uchar>(ny, nx);
                                float neighbor_depth = depth.at<float>(ny, nx);
                                // 邻居有效条件：同类背景标签且深度相近
                                if (neighbor_label == 1 &&
                                    neighbor_depth > 0 && neighbor_depth < 6.0f &&
                                    std::abs(neighbor_depth - d) < 0.5f)
                                {
                                    valid_neighbors++;
                                }
                            }
                        }
                    }
                    // 背景区域要求至少4个有效邻居（更严格）
                    if (valid_neighbors < 4)
                    {
                        continue;
                    }
                }

                // 障碍物像素施加单侧深度过滤：只滤掉明显在参考深度之后的拖尾点。
                if (pc_rgbl.label >= 100 &&
                    label_depth_reference.count(pc_rgbl.label) &&
                    d > label_depth_reference[pc_rgbl.label] + far_depth_tolerance)
                {
                    continue;
                }
                xyz_rgbl_cloud.push_back(pc_rgbl);
                total_points_added++;
            }
        }
    }

    // ========== DEBUG: 统计各label的采样情况 ==========
    std::map<uint8_t, int> label_pixel_count;       // label在图像中的像素数
    std::map<uint8_t, int> label_valid_depth_count; // label有有效深度的像素数
    std::map<uint8_t, int> label_sampled_count;     // label被采样的点数
    int label_depth_filtered_count = 0;             // 被深度一致性过滤掉的点数

    for (int y = 0; y < safe_rows; y++)
    {
        for (int x = 0; x < safe_cols; x++)
        {
            uint8_t label = lab.at<uchar>(y, x);
            if (label >= 100)
            {
                label_pixel_count[label]++;
                float d = depth.at<float>(y, x);
                if (d > 0 && d < 6.0f)
                {
                    label_valid_depth_count[label]++;
                }
            }
        }
    }

    // 针对障碍物区域（Label >= 100）进行密集采样补充，并施加深度一致性过滤
    // 深度一致性过滤：跳过深度偏离障碍物主体超过阈值的像素，消除拖影
    for (int y = 0; y < safe_rows; y += 1)
    {
        for (int x = 0; x < safe_cols; x += 1)
        {
            uint8_t label = lab.at<uchar>(y, x);

            // 只对障碍物区域进行密集采样
            if (label < 100)
                continue;

            // 跳过已经被步长4采样过的点
            if (y % 4 == 0 && x % 4 == 0)
                continue;

            float d = depth.at<float>(y, x);
            if (d <= 0 || d >= 6.0f)
                continue;

            // 深度一致性过滤：跳过偏离障碍物主体深度超过阈值的像素（消除拖影）
            if (label_depth_reference.count(label) &&
                d > label_depth_reference[label] + far_depth_tolerance)
            {
                label_depth_filtered_count++;
                continue;
            }

            // 邻域一致性检查：统计3x3窗口内的有效邻居数，过滤孤立噪点
            int valid_neighbors = 0;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dy == 0 && dx == 0)
                        continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < safe_rows && nx >= 0 && nx < safe_cols)
                    {
                        uint8_t neighbor_label = lab.at<uchar>(ny, nx);
                        float neighbor_depth = depth.at<float>(ny, nx);
                        // 邻居有效条件：同类标签（障碍物）且深度相近
                        if (neighbor_label >= 100 && neighbor_depth > 0 &&
                            neighbor_depth < 6.0f &&
                            std::abs(neighbor_depth - d) < 0.3f)
                        {
                            valid_neighbors++;
                        }
                    }
                }
            }
            // 要求至少2个有效邻居才添加点，过滤孤立噪点
            if (valid_neighbors < 2)
            {
                continue;
            }

            pcl::PointXYZRGBL pc_rgbl;
            pc_rgbl.x = (x - cx) * d / fx;
            pc_rgbl.y = (y - cy) * d / fy;
            pc_rgbl.z = d;

            Vec3b color = ori_mat.at<Vec3b>(y, x);
            uint8_t r = static_cast<uint8_t>(color[2]);
            uint8_t g = static_cast<uint8_t>(color[1]);
            uint8_t b = static_cast<uint8_t>(color[0]);

            uint32_t rgb_packed =
                ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            pc_rgbl.rgb = *reinterpret_cast<float *>(&rgb_packed);
            pc_rgbl.label = label;

            xyz_rgbl_cloud.push_back(pc_rgbl);
            obstacle_points_added++;
            total_points_added++;
            label_sampled_count[label]++;
        }
    }

    // ========== DEBUG: 输出各label的采样统计 ==========
    std::cout << "\n========== 障碍物采样诊断 ==========" << std::endl;
    for (const auto &pair : label_pixel_count)
    {
        uint8_t label = pair.first;
        int total_pixels = pair.second;
        int valid_depth_pixels = label_valid_depth_count[label];
        int sampled_points = label_sampled_count[label];

        std::cout << "Label " << static_cast<int>(label) << ":" << std::endl;
        std::cout << "  - 标签图像素数: " << total_pixels << std::endl;
        std::cout << "  - 有效深度像素: " << valid_depth_pixels
                  << " (" << (100.0 * valid_depth_pixels / total_pixels) << "%)" << std::endl;
        std::cout << "  - 采样点云数: " << sampled_points
                  << " (" << (100.0 * sampled_points / valid_depth_pixels) << "%)" << std::endl;
    }
    std::cout << "[DepthFilter] 深度一致性过滤剔除点数: " << label_depth_filtered_count << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    if (obstacle_points_added > 0)
    {
        std::cout << "[Dense sampling] 障碍物区域额外添加 " << obstacle_points_added
                  << " 个点 (总点数: " << total_points_added << ")" << std::endl;
    }

    stereo_point_ori_rgb_filter(xyz_rgbl_cloud, out_xyz_rgbl_cloud);

    out_xyz_rgbl_cloud.height = 1;
    out_xyz_rgbl_cloud.width = out_xyz_rgbl_cloud.points.size();
    out_xyz_rgbl_cloud.points.resize(out_xyz_rgbl_cloud.width *
                                     out_xyz_rgbl_cloud.height);
}

void StereoMultiMatch::stereo_point_ori_rgb_filter(
    pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
    pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud)
{
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr input_cloud =
        std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>(xyz_rgbl_cloud);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr output_cloud(
        new pcl::PointCloud<pcl::PointXYZRGBL>());

    if (input_cloud->empty())
    {
        cout << "Input cloud is empty, nothing to filter." << endl;
        out_xyz_rgbl_cloud.clear();
        return;
    }

    pcl::PassThrough<pcl::PointXYZRGBL> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-3.0, 0.5);
    pass.setNegative(false);
    pass.filter(*output_cloud);

    pcl::PassThrough<pcl::PointXYZRGBL> pass2;
    pass2.setInputCloud(output_cloud);
    pass2.setFilterFieldName("z");
    pass2.setFilterLimits(0.0, 5.0);
    pass2.setNegative(false);
    pass2.filter(*output_cloud);

    if (match_enable_ces_show)
    {
        // ========== 离群点过滤（组合优化版-方案4）==========
        int points_before_filter = output_cloud->size();
        int points_after_voxel = 0;
        int points_after_ror = 0;
        int points_after_label = 0;

        // ========== DEBUG: 统计过滤前各label的点数 ==========
        std::map<uint32_t, int> label_count_before;
        for (const auto &pt : output_cloud->points)
        {
            if (pt.label >= 100)
            {
                label_count_before[pt.label]++;
            }
        }

        // 步骤1：VoxelGrid快速降采样（CPU友好，大幅减少点数）
        pcl::VoxelGrid<pcl::PointXYZRGBL> vg;
        vg.setInputCloud(output_cloud);
        vg.setLeafSize(0.025f, 0.025f, 0.025f); // 2.5cm体素
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr downsampled(
            new pcl::PointCloud<pcl::PointXYZRGBL>);
        vg.filter(*downsampled);
        points_after_voxel = downsampled->size();

        // ========== DEBUG: 统计体素降采样后各label的点数 ==========
        std::map<uint32_t, int> label_count_after_voxel;
        for (const auto &pt : downsampled->points)
        {
            if (pt.label >= 100)
            {
                label_count_after_voxel[pt.label]++;
            }
        }

        // 步骤2：在降采样点云上做轻量级离群点移除（参数降低以适应降采样后的点云密度）
        // 但对于小球等小目标（label 104），使用更宽松的参数
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr obstacle_cloud(
            new pcl::PointCloud<pcl::PointXYZRGBL>);
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr other_cloud(
            new pcl::PointCloud<pcl::PointXYZRGBL>);

        // 分离小球点云和其他点云
        for (const auto &pt : downsampled->points)
        {
            if (pt.label == 104)
            {
                obstacle_cloud->push_back(pt);
            }
            else
            {
                other_cloud->push_back(pt);
            }
        }

        std::cout << "[ROR Filter] 分离点云: Label 104=" << obstacle_cloud->size()
                  << " 点, 其他=" << other_cloud->size() << " 点" << std::endl;

        // 对其他点云使用正常的离群点移除
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBL> ror;
        ror.setInputCloud(other_cloud);
        ror.setRadiusSearch(0.10);      // 从 8cm 增加到 10cm，更严格过滤夜间噪点
        ror.setMinNeighborsInRadius(8); // 从 6 增加到 8，要求更多邻居
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr filtered_other(
            new pcl::PointCloud<pcl::PointXYZRGBL>);
        ror.filter(*filtered_other);

        // 对小球点云使用更宽松的参数（或完全跳过过滤）
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr filtered_obstacle(
            new pcl::PointCloud<pcl::PointXYZRGBL>);
        if (obstacle_cloud->size() > 0)
        {
            if (obstacle_cloud->size() >= 10)
            {
                // 如果小球点数足够多，使用宽松的离群点移除
                pcl::RadiusOutlierRemoval<pcl::PointXYZRGBL> ror_obstacle;
                ror_obstacle.setInputCloud(obstacle_cloud);
                ror_obstacle.setRadiusSearch(0.15);      // 更大的搜索半径15cm
                ror_obstacle.setMinNeighborsInRadius(3); // 更少的邻居要求
                ror_obstacle.filter(*filtered_obstacle);
                std::cout << "[ROR Filter] Label 104 使用宽松过滤: "
                          << obstacle_cloud->size() << " -> " << filtered_obstacle->size() << " 点" << std::endl;
            }
            else
            {
                // 如果小球点数很少，完全跳过离群点移除
                *filtered_obstacle = *obstacle_cloud;
                std::cout << "[ROR Filter] Label 104 点数过少(" << obstacle_cloud->size()
                          << ")，跳过离群点过滤" << std::endl;
            }
        }

        // 合并过滤后的点云
        *output_cloud = *filtered_other;
        *output_cloud += *filtered_obstacle;
        points_after_ror = output_cloud->size();

        // ========== DEBUG: 统计离群点移除后各label的点数 ==========
        std::map<uint32_t, int> label_count_after_ror;
        for (const auto &pt : output_cloud->points)
        {
            if (pt.label >= 100)
            {
                label_count_after_ror[pt.label]++;
            }
        }

        // 步骤3：利用分割信息过滤（排除背景和未知标签）
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr label_filtered(
            new pcl::PointCloud<pcl::PointXYZRGBL>);
        label_filtered->reserve(output_cloud->size());
        for (const auto &pt : output_cloud->points)
        {
            if (pt.label > 0 &&
                pt.label <= 200)
            {
                // 保留有效标签（排除0=未知, 1=背景）
                label_filtered->push_back(pt);
            }
        }
        *output_cloud = *label_filtered;
        points_after_label = output_cloud->size();

        // ========== DEBUG: 输出过滤过程中各label的损失 ==========
        std::cout << "\n========== 点云过滤诊断 ==========" << std::endl;
        std::cout << "总点数变化: " << points_before_filter << " -> "
                  << points_after_voxel << " (体素) -> "
                  << points_after_ror << " (离群点) -> "
                  << points_after_label << " (标签)" << std::endl;

        for (const auto &pair : label_count_before)
        {
            uint32_t label = pair.first;
            int before = pair.second;
            int after_voxel = label_count_after_voxel[label];
            int after_ror = label_count_after_ror[label];

            std::cout << "\nLabel " << label << " 过滤过程:" << std::endl;
            std::cout << "  原始: " << before << " 点" << std::endl;
            std::cout << "  体素降采样后: " << after_voxel << " 点 (损失 "
                      << (before - after_voxel) << ", "
                      << (100.0 * (before - after_voxel) / before) << "%)" << std::endl;
            std::cout << "  离群点移除后: " << after_ror << " 点 (损失 "
                      << (after_voxel - after_ror) << ", "
                      << (after_voxel > 0 ? 100.0 * (after_voxel - after_ror) / after_voxel : 0) << "%)" << std::endl;

            if (after_ror == 0 && before > 0)
            {
                std::cout << "  ⚠️  警告: Label " << label << " 的所有点被过滤器移除!" << std::endl;
            }
        }
        std::cout << "========================================\n"
                  << std::endl;
    }

    output_cloud->width = output_cloud->points.size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;

    out_xyz_rgbl_cloud = *output_cloud;
}

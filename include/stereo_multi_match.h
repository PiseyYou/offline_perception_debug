#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <math.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"

#include "multiscale_filter.hpp"

#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <perception_common.h>

#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/search.h>
#include <pcl/memory.h>

using namespace std;
using namespace cv;

class StereoMultiMatch
{
public:
    double cx, cy, fx, fy;
    Mat Pl, Pr;
    Mat P2, Q;
    bool match_enable_ces_show=false;

    cv::Mat depth_;
    cv::Mat disparity_;

    void stereo_multi_param_init();

    Mat stereo_multi_process_depth(Mat &rectifyL, Mat &rectifyR);
    Mat stereo_multi_process_filter(Mat &disparity_, cv::Mat lab_dst, bool enable_height_filter_);

    cv::Mat stereo_multi_process(cv::Mat& rectifyL, cv::Mat& rectifyR, bool enable_height_filter_);

    void stereo_point_ori_rgb_filter(pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud, pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud);

    void stereo_process_pc_rgbl_depth(const Mat &depth, Mat &ori_mat, pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud, pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud);

    void stereo_process_pc_rgbl_dest(const Mat &depth, Mat &ori_mat, std::vector<Detection> &dect_src,
                                                       pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud, pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud);
    void stereo_process_pci_depth_rgb_seg_fusion(const Mat &depth, const Mat &lab, Mat &ori_mat,
                                                                   pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
                                                                   pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud);
    void stereo_process_pci_depth_rgb_seg_det_fusion(const Mat &depth, const Mat &lab, std::vector<Detection> &dect_src, Mat &ori_mat,
                                                                       pcl::PointCloud<pcl::PointXYZRGBL> &xyz_rgbl_cloud,
                                                                       pcl::PointCloud<pcl::PointXYZRGBL> &out_xyz_rgbl_cloud);
    void det_pc_rgb_label(Detection& det, const cv::Point& pt, pcl::PointXYZRGBL& pci);
    bool in_range(const cv::Point& top_left, const cv::Point& bottom_right, const cv::Point& pt_2d);

private:
    using Clock = std::chrono::high_resolution_clock;

    Mat stereoImg;
    Mat rgbImageL, rgbImageR;
    Mat grayImageL, grayImageR;
    Mat rectifyImageL, rectifyImageR;

    Mat half_grayImageL, half_grayImageR;
    Mat temp_grayImageL;

    bool use_background_substract_= false;
    bool use_multiscale_filter_= true;

    int intensity_low_ = 15;
    int intensity_high_ = 235;

    int bilateral_filter_kernel_size_= -1;
    double bilateral_filter_sigma_space_= 75.0;
    double bilateral_filter_sigma_color_= 75.0;
    int median_filter_kernel_size_= -1;

    int diff_threshold=2;
    int potential_threshold=6;
    int growing_window_size=5;
    int growing_threshold=2;
    int excessive_threshold=20;

    void stereo_block_matcher_init();
    void half_top_stereo_block_matcher_init();
    void half_bottom_stereo_block_matcher_init();
    void stereo_dis_init();

    void stereo_base_param_init();
    bool setStereoMatcherParameters(string dirPath);
    cv::Mat backgroundSubstract(const cv::Mat& src);


    // Variable
    cv::Ptr<cv::StereoBM> stereo_block_matcher_;
    cv::Ptr<cv::StereoBM> half_top_stereo_block_matcher_;
    cv::Ptr<cv::StereoSGBM> half_bottom_stereo_block_matcher_;
    cv::Mat disparity_16S_;

    cv::Mat disparity_half_16S_;
    cv::Mat disparity_half_;
    cv::Mat disparity_half_upscale_;
    cv::Mat intensity_mask_;
    cv::Mat disparity_valid_mask_;
    cv::Mat l_filtered_, r_filtered_;

    MultiScaleFilterParams orig_param_, half_param_;
};

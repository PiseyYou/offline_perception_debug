#include "unified_perception_processor.h"
#include "stereo_multi_match.h"
#include "cdt_perception.h"
#include "det_perception.h"
#include "seg_perception.h"
#include "qr_cs_perception.h"
#include "multi_sub_perception.h"
#include "dsg_perception.h"
#include "cls_perception.h"
#include "stereo_point_cloud_rgbl.h"
#include "offline_utils.hpp"

#include <chrono>
#include <iostream>
#include <algorithm>

namespace offline_perception {

UnifiedPerceptionProcessor::UnifiedPerceptionProcessor(const Config& cfg) : config_(cfg) {
    stereo_matcher_ = std::make_unique<StereoMultiMatch>();
    cdt_perceptor_ = std::make_unique<CdtPerception>();
    det_perceptor_ = std::make_unique<DetPerception>();
    seg_perceptor_ = std::make_unique<SegPerception>();
    qr_cs_perceptor_ = std::make_unique<QrCsPerception>();
    multi_perceptor_ = std::make_unique<MultiPerception>();
    sub_perceptor_ = std::make_unique<MulSubPerception>();
    dsg_perceptor_ = std::make_unique<DsgPerception>();
    dsg_multi_perceptor_ = std::make_unique<DsgMultiPerception>();
}

UnifiedPerceptionProcessor::~UnifiedPerceptionProcessor() = default;

bool UnifiedPerceptionProcessor::init() {
    std::cout << "Initializing UnifiedPerceptionProcessor in mode " << config_.infer_mode << std::endl;
    stereo_matcher_->stereo_multi_param_init();
    std::string base = config_.model_dir;
    cdt_perceptor_->perception_init((base + config_.cdt_model).c_str());
    switch (config_.infer_mode) {
        case 2: det_perceptor_->perception_init((base + config_.det_model).c_str(), config_.detection_threshold); break;
        case 3: seg_perceptor_->perception_init((base + config_.seg_model).c_str()); break;
        case 4: qr_cs_perceptor_->perception_init((base + config_.cs_model).c_str()); break;
        case 5: multi_perceptor_->perception_init((base + config_.multi_model).c_str()); break;
        case 6: sub_perceptor_->perception_init((base + config_.sub_model).c_str()); break;
        case 7:
            dsg_perceptor_->perception_init((base + config_.dsg_model).c_str());
            if (config_.enable_cdt) cdt_perceptor_->perception_init((base + config_.cdt_model).c_str());
            break;
        case 8: dsg_multi_perceptor_->perception_init((base + config_.dsg_multi_model).c_str()); break;
    }
    return true;
}

ProcessResult UnifiedPerceptionProcessor::process(const cv::Mat& left_img, const cv::Mat& right_img, int frame_id, const std::string& image_name) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat grayL, grayR;
    cv::cvtColor(left_img, grayL, cv::COLOR_BGR2GRAY);
    if (!right_img.empty()) cv::cvtColor(right_img, grayR, cv::COLOR_BGR2GRAY);
    ProcessResult res; res.frame_id = frame_id;
    switch (config_.infer_mode) {
        case 0: res = processMode0(left_img, grayL, grayR); break;
        case 1: res = processMode1(left_img, grayL, grayR); break;
        case 2: res = processMode2(left_img, grayL, grayR); break;
        case 3: res = processMode3(left_img, grayL, grayR); break;
        case 4: res = processMode4(left_img, grayL, grayR); break;
        case 5: res = processMode5(left_img, grayL, grayR); break;
        case 6: res = processMode6(left_img, grayL, grayR); break;
        case 7: res = processMode7(left_img, grayL, grayR); break;
        case 8: res = processMode8(left_img, grayL, grayR); break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    res.process_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode1(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "DepthOnly";
    if (grayR.empty()) return res;
    cv::Mat disparity = stereo_matcher_->stereo_multi_process_depth(grayL, grayR);
    res.depth_map = stereo_matcher_->stereo_multi_process_filter(disparity, cv::Mat(), config_.enable_height_filter);
    cv::Mat left_copy = left.clone();
    stereo_matcher_->stereo_process_pc_rgbl_depth(res.depth_map, left_copy, res.point_cloud, res.point_cloud);
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode2(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "Detection";
    std::vector<Detection> detections; cv::Mat left_copy = left.clone();
    det_perceptor_->perception_process(left_copy);
    det_perceptor_->perception_postprocess_nanodet(left_copy, detections);
    det_perceptor_->task_release();
    if (grayR.empty()) return res;
    cv::Mat disparity = stereo_matcher_->stereo_multi_process_depth(grayL, grayR);
    res.depth_map = stereo_matcher_->stereo_multi_process_filter(disparity, cv::Mat(), config_.enable_height_filter);
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud;
    stereo_matcher_->stereo_process_pc_rgbl_dest(res.depth_map, left_copy, detections, xyz_rgbl_cloud, res.point_cloud);
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode3(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "Segmentation";
    cv::Mat left_copy = left.clone();
    res.label_map = seg_perceptor_->perception_postprocess_int64_erode(left_copy, config_.erode_pixel);
    if (grayR.empty()) return res;
    cv::Mat disparity = stereo_matcher_->stereo_multi_process_depth(grayL, grayR);
    res.depth_map = stereo_matcher_->stereo_multi_process_filter(disparity, res.label_map, config_.enable_height_filter);
    pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud;
    stereo_matcher_->stereo_process_pci_depth_rgb_seg_fusion(res.depth_map, res.label_map, left_copy, xyz_rgbl_cloud, res.point_cloud);
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode5(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "MultiTask";
    cv::Rect cropRegion(0, 0, left.cols, std::min(left.rows, 384));
    cv::Mat croppedImg = left(cropRegion);
    std::vector<Detection> det_raw, det_dst, cdt_src; cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 1; cv::Rect cdt_rect;
    if (config_.enable_cdt) { cdt_perceptor_->perception_process_bgr(croppedImg, cdt_src); cdt_rect = get_cdt_rect(cdt_src); }
    multi_perceptor_->perception_process_bgr_no_argmax_erode_mul(croppedImg, det_raw, img_label, config_.erode_pixel);
    cv::Mat lab_dst; img_label.copyTo(lab_dst);
    for (const auto& det : det_raw) { Detection fd = det; fd.id += 100; det_dst.push_back(fd); }
    bool all_lawn = false; processLabelContours(lab_dst, all_lawn);
    if (config_.enable_cdt && cdt_rect.area() > 0) lab_dst(cdt_rect) = -1;
    res.label_map = lab_dst.clone();
    if (grayR.empty()) return res;
    cv::Mat disparity = stereo_matcher_->stereo_multi_process_depth(grayL, grayR);
    res.depth_map = stereo_matcher_->stereo_multi_process_filter(disparity, res.label_map, config_.enable_height_filter);
    if (res.depth_map.rows > croppedImg.rows) res.depth_map = res.depth_map(cv::Rect(0, 0, res.depth_map.cols, croppedImg.rows)).clone();
    pcl::PointCloud<pcl::PointXYZRGBL> cloud_tmp; stereo_matcher_->stereo_process_pci_depth_rgb_seg_det_fusion(res.depth_map, res.label_map, det_dst, croppedImg, cloud_tmp, res.point_cloud);
    if (config_.enable_debug_show) {
        cv::Mat img_seg_show, xyz_rgbl_show, origin_seg, pure_seg;
        pure_seg = drawResultOptimized(croppedImg, res.label_map, det_dst, img_seg_show);
        cv::hconcat(croppedImg, pure_seg, origin_seg);
        cv::hconcat(origin_seg, img_seg_show, origin_seg);
        stereo_point_cloud spc;
        spc.show_xyz_rgbl_plane_point_cloud_final(res.point_cloud, xyz_rgbl_show);
        cv::vconcat(origin_seg, xyz_rgbl_show, res.visualization);
    }
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode6(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "SubTask";
    cv::Mat croppedImg = (left.rows == 480) ? left(cv::Rect(0,0,640,384)) : left;
    std::vector<Detection> det_raw, det_dst, cdt_src; cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2; cv::Mat lab_out;
    cv::Rect cdt_rect; if (config_.enable_cdt) { cdt_perceptor_->perception_process_bgr(croppedImg, cdt_src); cdt_rect = get_cdt_rect(cdt_src); }
    sub_perceptor_->perception_process_bgr_no_argmax_erode(croppedImg, det_raw, img_label, lab_out, config_.erode_pixel);
    cv::Mat lab_dst; lab_out.copyTo(lab_dst);
    for (const auto& det : det_raw) { Detection fd = det; fd.id += 100; det_dst.push_back(fd); }
    bool all_lawn = false; processLabelContours(lab_dst, all_lawn);
    lab_dst(cdt_rect) = -1; res.label_map = lab_dst.clone();
    if (grayR.empty()) return res;
    cv::Mat disparity = stereo_matcher_->stereo_multi_process_depth(grayL, grayR);
    res.depth_map = stereo_matcher_->stereo_multi_process_filter(disparity, res.label_map, config_.enable_height_filter);
    if (res.depth_map.rows > croppedImg.rows) res.depth_map = res.depth_map(cv::Rect(0, 0, res.depth_map.cols, croppedImg.rows)).clone();
    pcl::PointCloud<pcl::PointXYZRGBL> cloud_tmp; stereo_matcher_->stereo_process_pci_depth_rgb_seg_det_fusion(res.depth_map, res.label_map, det_dst, croppedImg, cloud_tmp, res.point_cloud);
    if (config_.enable_debug_show) {
        cv::Mat img_seg_show, xyz_rgbl_show, origin_seg, pure_seg;
        pure_seg = drawResultOptimized(croppedImg, res.label_map, det_dst, img_seg_show);
        cv::hconcat(croppedImg, pure_seg, origin_seg);
        cv::hconcat(origin_seg, img_seg_show, origin_seg);
        stereo_point_cloud spc;
        spc.show_xyz_rgbl_plane_point_cloud_final(res.point_cloud, xyz_rgbl_show);
        cv::vconcat(origin_seg, xyz_rgbl_show, res.visualization);
    }
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode7(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "DSG_Night";
    // 裁剪 640x432（对齐参考脚本 run_cdt_dsg_fusion_dir.cpp）
    cv::Mat croppedImg = (left.rows == 480) ? left(cv::Rect(0, 0, 640, 432)) : left.clone();

    // resize 到 640x384 进行推理
    cv::Mat img384; cv::resize(croppedImg, img384, cv::Size(640, 384));

    // 初始化 img_label 为背景类别=1（与参考脚本一致：Mat::zeros + 1）
    cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 1;
    cv::Mat lab_out;

    // 多任务推理：使用 perception_process_bgr_no_argmax_erode 获取多类别分割 + 检测结果
    // 参考脚本使用此接口，process_infer_match 仅返回全1（单类别背景），不可用
    std::vector<Detection> dect_src;
    dsg_perceptor_->ori_height = img384.rows;
    dsg_perceptor_->ori_width  = img384.cols;
    dsg_perceptor_->perception_process_bgr_no_argmax_erode(
        img384, dect_src, img_label, lab_out, config_.erode_pixel);

    // 将分割结果从 640x384 resize 回 640x432
    cv::resize(lab_out, lab_out, cv::Size(640, 432), 0, 0, cv::INTER_NEAREST);

    // 构建 lab_dst，并将检测 id 映射为 100+ 格式（与参考脚本一致）
    cv::Mat lab_dst;
    lab_out.copyTo(lab_dst);
    std::vector<Detection> dect_dst;
    for (const auto& det : dect_src) {
        Detection fd = det; fd.id += 100; dect_dst.push_back(fd);
    }

    // CES：处理被草坪包围的区域（标记为 class 13）
    bool all_lawn = false;
    processLabelContours(lab_dst, all_lawn);

    // CDT 处理（与参考脚本顺序一致：CES 之后、fusion 之前）
    if (config_.enable_cdt) {
        std::vector<Detection> cdt_src;
        cdt_perceptor_->perception_process_bgr(img384, cdt_src);
        cv::Rect cdt_rect = get_cdt_rect(cdt_src);
        if (cdt_rect.area() > 0) lab_dst(cdt_rect) = -1;
    }

    res.label_map = lab_dst.clone();

    if (grayR.empty()) return res;

    // 深度计算：使用 stereo_multi_process（与参考脚本一致，不加 height_filter）
    cv::Mat depth_cal = stereo_matcher_->stereo_multi_process(grayL, grayR, config_.enable_height_filter);
    cv::Mat depth432 = (depth_cal.rows >= 432) ? depth_cal(cv::Rect(0, 0, 640, 432)).clone() : depth_cal;

    res.depth_map = depth432;

    // Fusion：与参考脚本使用相同接口，传入检测框
    pcl::PointCloud<pcl::PointXYZRGBL> cloud_tmp;
    stereo_matcher_->stereo_process_pci_depth_rgb_seg_det_fusion(
        depth432, lab_dst, dect_dst, croppedImg, cloud_tmp, res.point_cloud);

    res.label_map = lab_dst;

    if (config_.enable_debug_show) {
        cv::Mat img_seg_show, xyz_rgbl_show, origin_seg, pure_seg;
        cv::Mat croppedImgVis; cv::resize(croppedImg, croppedImgVis, cv::Size(640, 432));
        cv::Mat lab_vis; cv::resize(lab_dst, lab_vis, cv::Size(640, 432), 0, 0, cv::INTER_NEAREST);
        pure_seg = drawResultOptimized(croppedImgVis, lab_vis, dect_dst, img_seg_show);
        cv::hconcat(croppedImgVis, pure_seg, origin_seg);
        cv::hconcat(origin_seg, img_seg_show, origin_seg);
        stereo_point_cloud spc;
        spc.show_xyz_rgbl_plane_point_cloud_final(res.point_cloud, xyz_rgbl_show);
        cv::vconcat(origin_seg, xyz_rgbl_show, res.visualization);
    }
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode8(const cv::Mat& left, cv::Mat& grayL, cv::Mat& grayR) {
    ProcessResult res; res.mode_name = "DSG_Multi";
    cv::Rect cropRegion(0,0,left.cols,384); cv::Mat croppedImg = left(cropRegion);
    std::vector<Detection> cdt_src, empty_det; cv::Mat lab_out; if (config_.enable_cdt) cdt_perceptor_->perception_process_bgr(croppedImg, cdt_src);
    dsg_multi_perceptor_->process_infer_match(croppedImg, lab_out);
    bool all_lawn = false; processLabelContours(lab_out, all_lawn);
    cv::Rect cdt_rect = get_cdt_rect(cdt_src); if (config_.enable_cdt && cdt_rect.area() > 0) lab_out(cdt_rect) = -1;
    res.label_map = lab_out.clone();
    if (grayR.empty()) return res;
    cv::Mat disp = stereo_matcher_->stereo_multi_process(grayL, grayR, config_.enable_height_filter);
    res.depth_map = disp(cv::Rect(0,0,640,384)).clone();
    stereo_matcher_->stereo_process_pci_depth_rgb_seg_det_fusion(res.depth_map, lab_out, empty_det, croppedImg, res.point_cloud, res.point_cloud);
    if (config_.enable_debug_show) {
        cv::Mat img_seg_show, xyz_rgbl_show, origin_seg;
        drawResultOptimized(croppedImg, lab_out, empty_det, img_seg_show);
        cv::hconcat(croppedImg, img_seg_show, origin_seg);
        stereo_point_cloud spc;
        spc.show_xyz_rgbl_plane_point_cloud_final(res.point_cloud, xyz_rgbl_show);
        if (origin_seg.cols != xyz_rgbl_show.cols || origin_seg.type() != xyz_rgbl_show.type()) {
            cv::resize(origin_seg, origin_seg, cv::Size(xyz_rgbl_show.cols, 384));
        }
        cv::vconcat(origin_seg, xyz_rgbl_show, res.visualization);
    }
    return res;
}

ProcessResult UnifiedPerceptionProcessor::processMode0(const cv::Mat& l, cv::Mat& gL, cv::Mat& gR) { return {}; }
ProcessResult UnifiedPerceptionProcessor::processMode4(const cv::Mat& l, cv::Mat& gL, cv::Mat& gR) { return processMode1(l, gL, gR); }

float UnifiedPerceptionProcessor::computeStraightEdgeScore(const cv::Mat &grayROI, const cv::Mat &compMask) {
    cv::Mat edges; cv::Canny(grayROI, edges, 50, 150); edges &= compMask;
    if (cv::countNonZero(edges) < 10) return 0.0f;
    std::vector<cv::Vec4i> lines; cv::HoughLinesP(edges, lines, 1, CV_PI/180, 15, 10, 5);
    if (lines.empty()) return 0.0f;
    cv::Mat lm = cv::Mat::zeros(edges.size(), CV_8UC1);
    for (const auto &l : lines) cv::line(lm, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 255, 2);
    return static_cast<float>(cv::countNonZero(lm & edges)) / cv::countNonZero(edges);
}

void UnifiedPerceptionProcessor::refineObstacleByColorAndEdge(cv::Mat &label, const cv::Mat &bgrImg, int min_area, float min_edge_score, float min_solidity, float min_green_ratio) {
    if (label.empty() || bgrImg.empty()) return;
    cv::Mat blurred, hsv, sg; cv::GaussianBlur(bgrImg, blurred, cv::Size(5, 5), 0); cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(25, 40, 40), cv::Scalar(85, 255, 255), sg);
    cv::Mat roi_g = sg(cv::Range(label.rows*0.5, label.rows*0.9), cv::Range::all());
    if ((float)cv::countNonZero(roi_g)/roi_g.total() < min_green_ratio) return;
    cv::Mat veg, nveg; cv::inRange(hsv, cv::Scalar(15, 30, 40), cv::Scalar(85, 255, 255), veg); cv::bitwise_not(veg, nveg);
    cv::Mat h[3]; cv::split(hsv, h); nveg &= ((h[1]>25 | h[2]>150) & h[2]>40);
    cv::Mat l2, l3, cand = nveg & (cv::Mat(label==2) | cv::Mat(label==3));
    cand(cv::Range(0, label.rows*0.5), cv::Range::all()).setTo(0); cand(cv::Range(label.rows*0.9, label.rows), cv::Range::all()).setTo(0);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)); cv::morphologyEx(cand, cand, cv::MORPH_OPEN, k); cv::morphologyEx(cand, cand, cv::MORPH_CLOSE, k);
    cv::Mat lbs, sts, ctrs; int n = cv::connectedComponentsWithStats(cand, lbs, sts, ctrs, 8, CV_32S); cv::Mat gray; cv::cvtColor(bgrImg, gray, cv::COLOR_BGR2GRAY);
    for(int i=1; i<n; ++i) {
        if (sts.at<int>(i, cv::CC_STAT_AREA) < min_area) continue;
        cv::Rect r(sts.at<int>(i, cv::CC_STAT_LEFT), sts.at<int>(i, cv::CC_STAT_TOP), sts.at<int>(i, cv::CC_STAT_WIDTH), sts.at<int>(i, cv::CC_STAT_HEIGHT));
        cv::Mat m = (lbs(r) == i); float es = computeStraightEdgeScore(gray(r), m), sol = 0.0f;
        std::vector<std::vector<cv::Point>> c; cv::findContours(m.clone(), c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if(!c.empty()) {
            auto& cnt = *std::max_element(c.begin(), c.end(), [](auto& a, auto& b){return cv::contourArea(a)<cv::contourArea(b);});
            std::vector<cv::Point> hl; cv::convexHull(cnt, hl); if(cv::contourArea(hl)>0) sol = cv::contourArea(cnt)/cv::contourArea(hl);
        }
        if (es >= ((sts.at<int>(i, cv::CC_STAT_AREA) > min_area*5) ? min_edge_score*0.6f : min_edge_score) && sol >= min_solidity) label.setTo(5, lbs==i);
    }
}

void UnifiedPerceptionProcessor::applyDarknessFilter(cv::Mat &label, const cv::Mat &bgrImg, int darkness_threshold) {
    if (label.empty() || bgrImg.empty() || label.size() != bgrImg.size()) return;
    for (int y = 0; y < label.rows; ++y) {
        const cv::Vec3b *bgr_row = bgrImg.ptr<cv::Vec3b>(y);
        uchar *label_row = label.ptr<uchar>(y);
        for (int x = 0; x < label.cols; ++x) {
            const cv::Vec3b &px = bgr_row[x];
            if (px[0] <= darkness_threshold && px[1] <= darkness_threshold && px[2] <= darkness_threshold) label_row[x] = 2;
        }
    }
}

void UnifiedPerceptionProcessor::filterNighttimeDepth(cv::Mat &depth, const cv::Mat &label_map) {
    if (depth.empty()) return;
    cv::Mat k5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat k7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat tmp; cv::morphologyEx(depth, tmp, cv::MORPH_CLOSE, k5);
    cv::Mat df = tmp.clone();
    if (tmp.rows > 300) { cv::Mat ri = tmp(cv::Rect(0, 300, tmp.cols, tmp.rows - 300)); cv::Mat ro = df(cv::Rect(0, 300, tmp.cols, tmp.rows - 300)); cv::GaussianBlur(ri, ro, cv::Size(11, 1), 0); }
    cv::Mat vm; cv::threshold(df, vm, 0, 255, cv::THRESH_BINARY); vm.convertTo(vm, CV_8UC1);
    cv::Mat om; cv::morphologyEx(vm, om, cv::MORPH_OPEN, k7); df.setTo(0, om == 0); depth = df;
}

void UnifiedPerceptionProcessor::filterNighttimePointCloud(pcl::PointCloud<pcl::PointXYZRGBL> &cloud, const cv::Mat &label_map) {
    if (cloud.empty()) return;
    const float inv_grid = 10.0f; auto encode = [inv_grid](float x, float y, float z) -> int64_t { return static_cast<int64_t>(std::floor(x * inv_grid)) * 1000000LL + static_cast<int64_t>(std::floor(y * inv_grid)) * 1000LL + static_cast<int64_t>(std::floor(z * inv_grid)); };
    std::unordered_map<int64_t, std::vector<size_t>> grid; grid.reserve(cloud.size());
    for (size_t i = 0; i < cloud.size(); i++) { const auto &pt = cloud[i]; if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) grid[encode(pt.x, pt.y, pt.z)].push_back(i); }
    pcl::PointCloud<pcl::PointXYZRGBL> filtered; filtered.reserve(cloud.size());
    for (const auto &[key, indices] : grid) { if (indices.size() >= 3) for (size_t idx : indices) filtered.push_back(cloud[idx]); }
    cloud = std::move(filtered);
}

void UnifiedPerceptionProcessor::processLabelContours(cv::Mat &label_img, bool &all_neighbors_are_lawn) {
    if (label_img.empty() || cv::countNonZero(label_img == 2) == 0) return;
    cv::Mat result = label_img.clone(); cv::Mat mnt = (label_img != 0) & (label_img != 1) & (label_img != 2);
    std::vector<std::vector<cv::Point>> contours; cv::findContours(mnt, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1}; int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    for (const auto &contour : contours) {
        cv::Mat rm = cv::Mat::zeros(label_img.size(), CV_8UC1); cv::fillPoly(rm, {contour}, cv::Scalar(255));
        int lnc = 0, l1nc = 0, tnc = 0; bool touch = false;
        for (const auto &pt : contour) {
            if (pt.x == 0 || pt.x == label_img.cols - 1 || pt.y == 0 || pt.y == label_img.rows - 1) touch = true;
            for (int i = 0; i < 8; i++) {
                int nx = pt.x + dx[i], ny = pt.y + dy[i];
                if (nx >= 0 && nx < label_img.cols && ny >= 0 && ny < label_img.rows && rm.at<uchar>(ny, nx) == 0) {
                    tnc++; uchar nl = label_img.at<uchar>(ny, nx); if (nl == 2) lnc++; else if (nl == 1) l1nc++;
                }
            }
        }
        if (l1nc == 0 && (tnc == 0 || (float)lnc / tnc >= 0.8f || touch)) {
            for (int y = 0; y < label_img.rows; y++) for (int x = 0; x < label_img.cols; x++) if (rm.at<uchar>(y, x) == 255 && label_img.at<uchar>(y, x) > 2) result.at<uchar>(y, x) = 13;
            all_neighbors_are_lawn = true;
        }
    }
    label_img = result;
}

void UnifiedPerceptionProcessor::filterLabelDect(cv::Mat& src_lab, std::vector<Detection>& dect_src, cv::Mat& lab_dst, std::vector<Detection>& dect_dst, bool enable_det) {
    src_lab.copyTo(lab_dst); cv::Mat mz; cv::compare(lab_dst, 0, mz, cv::CMP_EQ); lab_dst.setTo(2, mz);
    for(size_t i=0;i<dect_src.size();++i) {
        Bbox b = dect_src[i].bbox; int8_t tid = dect_src[i].id;
        int x1=std::max((int)std::lround(b.xmin),0), y1=std::max((int)std::lround(b.ymin),0), x2=std::min((int)std::lround(b.xmax),lab_dst.cols-1), y2=std::min((int)std::lround(b.ymax),lab_dst.rows-1);
        if(x1>=x2||y1>=y2) continue; cv::Mat roi = lab_dst(cv::Rect(x1,y1,x2-x1,y2-y1));
        if(tid==3) { cv::Mat m1,m5; cv::inRange(roi,1,1,m1); cv::inRange(roi,5,5,m5); roi.setTo(103,m1|m5); }
        else if(tid==6) { cv::Mat m1,m5; cv::inRange(roi,1,1,m1); cv::inRange(roi,5,5,m5); roi.setTo(106,m1|m5); }
        else if(tid==4) {
            int cb=cv::countNonZero(roi==1), cw=cv::countNonZero((roi==2)|(roi==3));
            if(cb==0 && cw>0) { roi.setTo(104); dect_dst.push_back(dect_src[i]); }
            else if (!(cb>0 && cw>0)) { cv::Mat m1,m5; cv::inRange(roi,1,1,m1); cv::inRange(roi,5,5,m5); roi.setTo(104,m1|m5); dect_dst.push_back(dect_src[i]); }
        } else if(enable_det) { roi.setTo(tid+100); dect_dst.push_back(dect_src[i]); }
    }
}

cv::Rect UnifiedPerceptionProcessor::get_cdt_rect(const std::vector<Detection>& detections) {
    if(detections.size()!=1) return cv::Rect(); const auto& d = detections[0];
    cv::Rect r((int)d.bbox.xmin, (int)d.bbox.ymin, (int)(d.bbox.xmax-d.bbox.xmin), config_.image_height-(int)d.bbox.ymin);
    if(r.x>=0 && r.y>=0 && r.x+r.width<=config_.image_width && r.y+r.height<=config_.image_height) return r;
    return cv::Rect();
}

cv::Mat UnifiedPerceptionProcessor::drawResultOptimized(const cv::Mat& img, const cv::Mat& label, const std::vector<Detection>& dect, cv::Mat& img_seg_show, bool enable_draw_box) {
    static const std::array<cv::Vec3b, 256> lut = []() { std::array<cv::Vec3b, 256> l; l.fill(cv::Vec3b(0,0,0)); l[1]={200,0,0}; l[2]={102,255,100}; l[3]={0,89,118}; l[4]={0,255,255}; l[5]={0,0,255}; l[103]={255,0,255}; l[104]={0,0,255}; l[106]={255,255,0}; return l; }();
    cv::Mat p(label.size(), CV_8UC3); for(int i=0;i<label.rows;++i) { const uchar* r=label.ptr<uchar>(i); cv::Vec3b* o=p.ptr<cv::Vec3b>(i); for(int j=0;j<label.cols;++j) o[j]=lut[r[j]]; }
    if(p.size()!=img.size()) cv::resize(p, p, img.size(), 0, 0, cv::INTER_NEAREST);
    cv::addWeighted(img, 0.6f, p, 0.4f, 0.0, img_seg_show);
    if (enable_draw_box) {
        for(const auto& d : dect) { cv::rectangle(img_seg_show, cv::Rect(d.bbox.xmin, d.bbox.ymin, d.bbox.xmax-d.bbox.xmin, d.bbox.ymax-d.bbox.ymin), cv::Scalar(0,255,0), 2);
            cv::putText(img_seg_show, "id_"+std::to_string(d.id), cv::Point(d.bbox.xmin, std::max((int)d.bbox.ymin+15, 15)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1); }
    }
    return p;
}

} // namespace offline_perception

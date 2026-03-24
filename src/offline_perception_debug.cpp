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

// 离线感知处理器 - 支持多种模式
class OfflinePerceptionProcessor
{
public:
    // 配置参数
    struct Config
    {
        int infer_mode = 5; // 推理模式 0-6
        int erode_pixel = 205; // 腐蚀像素
        float area_threshold = 0.5; // 区域阈值
        float detection_threshold = 0.51; // 检测阈值
        float m_area_threshold = 0.5;
        bool enable_height_filter_ = false;
        bool enabel_cdt = true;
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
        // string sub_model = "../models/sub_20260211_640x384.bin";
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

    OfflinePerceptionProcessor(const Config& cfg) : config_(cfg)
    {
    }

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

        cout << "===================================================\n" << endl;
        return true;
    }

    ProcessResult process(const Mat& left_img, const Mat& right_img, int frame_id,
                          const string& image_name = "")
    {
        auto start_time = chrono::high_resolution_clock::now();

        ProcessResult result;
        result.frame_id = frame_id;
        current_frame_id_ = frame_id; // 保存当前帧ID
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
    int current_frame_id_ = 0; // 当前处理的帧ID
    string current_image_name_ = ""; // 当前处理的图像文件名（不含扩展名）

    // 画出depth_cal在原图上的有效矩形边界区域，保存图片并输出矩形信息
    cv::Rect drawDepthValidRegion(const Mat& depth_cal, const Mat& ori_img,
                                  const string& save_dir,
                                  const string& image_name)
    {
        // 创建有效深度的二值mask: 有效深度为 depth > 0 且 depth < 10.0 (MAX_DEPTH)
        Mat valid_mask;
        cv::Mat mask1 = (depth_cal > 0.01); // 排除无效的0值
        cv::Mat mask2 = (depth_cal < 9.99); // 排除被设为MAX_DEPTH(10.0)的无效值
        cv::bitwise_and(mask1, mask2, valid_mask);
        valid_mask.convertTo(valid_mask, CV_8U);

        // 查找有效区域的最小外接矩形
        cv::Rect valid_rect = cv::boundingRect(valid_mask);

        // 在原图上画出有效矩形边界
        Mat draw_img = ori_img.clone();
        if (valid_rect.area() > 0)
        {
            cv::rectangle(draw_img, valid_rect, Scalar(0, 255, 0), 2); // 绿色矩形框
            // 在矩形上方标注信息
            string rect_info = "Valid Depth Region [" + to_string(valid_rect.x) +
                "," + to_string(valid_rect.y) + " " +
                to_string(valid_rect.width) + "x" +
                to_string(valid_rect.height) + "]";
            cv::putText(draw_img, rect_info,
                        Point(valid_rect.x, max(valid_rect.y - 10, 15)),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        // 保存图片
        string save_path = save_dir + image_name + "_depth_valid_region.jpg";
        imwrite(save_path, draw_img);

        // 输出有效矩形框信息
        cout << "===== Depth Valid Region Info =====" << endl;
        cout << "  Image: " << image_name << endl;
        cout << "  Depth size: " << depth_cal.cols << "x" << depth_cal.rows << endl;
        cout << "  Valid rect: [x=" << valid_rect.x << ", y=" << valid_rect.y
            << ", w=" << valid_rect.width << ", h=" << valid_rect.height << "]"
            << endl;
        cout << "  Valid rect area: " << valid_rect.area() << " pixels" << endl;
        int total_pixels = depth_cal.cols * depth_cal.rows;
        int valid_pixels = cv::countNonZero(valid_mask);
        cout << "  Valid pixels: " << valid_pixels << " / " << total_pixels << " ("
            << fixed << setprecision(1) << (100.0 * valid_pixels / total_pixels)
            << "%)" << endl;
        cout << "  Saved: " << save_path << endl;
        cout << "==================================" << endl;

        return valid_rect;
    }

    // Mode 0: Disable - 不进行任何处理
    ProcessResult processMode0(const Mat& left, const Mat& right,
                               const Mat& grayImageL, const Mat& grayImageR)
    {
        ProcessResult result;
        result.mode_name = "Disable";
        cout << "[Mode 0] Disable - no processing" << endl;
        return result;
    }

    // Mode 1: Only Depth - 仅深度，无DL模型
    ProcessResult processMode1(const Mat& left, const Mat& right, Mat grayImageL,
                               Mat grayImageR)
    {
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
    ProcessResult processMode2(const Mat& left, const Mat& right, Mat grayImageL,
                               Mat grayImageR)
    {
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
    ProcessResult processMode3(const Mat& left, const Mat& right, Mat grayImageL,
                               Mat grayImageR)
    {
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
    ProcessResult processMode4(const Mat& left, const Mat& right, Mat grayImageL,
                               Mat grayImageR)
    {
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
    ProcessResult processMode5(const Mat& left, const Mat& right, Mat grayImageL,
                               Mat grayImageR)
    {
        ProcessResult result;
        result.mode_name = "MultiTask";

        cout << "[Mode 5] Multi-task recognition (seg + det)" << endl;

        // Step 1: 多任务推理
        auto infer_start = chrono::high_resolution_clock::now();
        // cv::Mat croppedImg = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
        // std::vector<Detection> detections;
        // Mat left_copy = left.clone();

        cv::Rect cropRegion(0, 0, left.cols, 384);
        cv::Mat croppedImg = left(cropRegion);

        cv::Mat dst_label(croppedImg.rows, croppedImg.cols, CV_8UC1);
        std::vector<Detection> detections, dect_dst, ct_dect_src;
        cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
        Mat lab_out, lab_temp;

        Rect cdt_rect;
        // cout << "count_stereo % frq_cdt: " << count_stereo % frq_cdt << endl;
        // if(config_.enabel_cdt && i % config_.frq_cdt == 0){
        if (config_.enabel_cdt)
        {
            ct_dect_src.clear();
            cdtPerception_.perception_process_bgr(croppedImg, ct_dect_src);
            // cout << "ct_dect_src size: " << ct_dect_src.size() << endl;
            cdt_rect = get_cdt_rect(ct_dect_src);
            // RCLCPP_INFO(this->get_logger(), "[global]===depth_time_cost:
            // %.2f===reciev_latency: %.2f", depth_time_cost, depth_reciev_latency);
            cout << "cdt_rect: " << cdt_rect << endl;
            cout << "cdt_rect.area: " << cdt_rect.area() << endl;
            // RCLCPP_INFO(this->get_logger(), "cdt_rect: [x=%d, y=%d, w=%d, h=%d],
            // area=%d", cdt_rect.x, cdt_rect.y, cdt_rect.width, cdt_rect.height,
            // static_cast<int>(cdt_rect.area())); ret_cdt=1;
        }

        // 优化1: 模型推理计时
        // double infer_start = rclcpp::Clock().now().seconds();
        multiPerception_.perception_process_bgr_no_argmax_erode_mul(
            croppedImg, detections, img_label, config_.erode_pixel);
        filterLabelDect(img_label, detections, lab_temp, dect_dst);
        lab_temp(cdt_rect) = -1;
        // dst_label = processLabelsOptimizedPipeline(lab_temp, croppedImg,
        //                                            config_.m_area_threshold);

        // printLabelDistribution(img_label);  // 直接打印分布

        auto infer_end = chrono::high_resolution_clock::now();
        cout << "[Step 1/4] Multi-task inference done: "
            << chrono::duration<double, milli>(infer_end - infer_start).count()
            << " ms" << endl;
        // cout << "  - Detections: " << detections.size() << " objects" << endl;

        // Step 2: 标签后处理 (简化版，跳过复杂的label processing)
        result.label_map = dst_label.clone();
        cout << "[Step 2/4] Label processing done (simplified)" << endl;

        // Step 3: 深度计算
        auto depth_start = chrono::high_resolution_clock::now();
        pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud, out_xyz_rgbl_cloud;
        Mat disparity =
            stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
        Mat depth_cal = stereo_multi_match.stereo_multi_process_filter(
            disparity, dst_label, config_.enable_height_filter_);

        // 深度图(640x480)与分割图/原图(640x384)尺寸不匹配，裁剪深度图上部384行对齐
        if (depth_cal.rows > croppedImg.rows)
        {
            depth_cal =
                depth_cal(cv::Rect(0, 0, depth_cal.cols, croppedImg.rows)).clone();
        }

        auto depth_end = chrono::high_resolution_clock::now();
        cout << "[Step 3/4] Depth computation done: "
            << chrono::duration<double, milli>(depth_end - depth_start).count()
            << " ms" << endl;

        // Step 4: 融合
        auto fusion_start = chrono::high_resolution_clock::now();

        stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
            depth_cal, result.label_map, detections, croppedImg, xyz_rgbl_cloud,
            out_xyz_rgbl_cloud);

        if (config_.m_enable_debug_show)
        {
            Mat img_seg_show;
            Mat pure_seg_mat =
                drawResult(croppedImg, dst_label, dect_dst, img_seg_show);
            Mat origin_seg;
            cv::resize(pure_seg_mat, pure_seg_mat, Size(640, 384));
            cv::resize(img_seg_show, img_seg_show, Size(640, 384));

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
    ProcessResult processMode6(const Mat& left, const Mat& right, Mat grayImageL,
                               Mat grayImageR)
    {
        ProcessResult result;
        result.mode_name = "SubTask";

        cout << "[Mode 6] Sub-task recognition" << endl;

        // Step 1: 子任务推理
        auto infer_start = chrono::high_resolution_clock::now();

        cv::Mat croppedImg;
        if (left.rows == 480)
        {
            cv::Rect cropRegion(0, 0, left.cols, 384);
            croppedImg = left(cropRegion);
        }
        else if (left.rows == 384)
        {
            croppedImg = left;
        }

        cv::Mat dst_label(croppedImg.rows, croppedImg.cols, CV_8UC1);
        std::vector<Detection> detections, dect_dst, ct_dect_src;
        cv::Mat img_label = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
        Mat lab_out, lab_temp;

        Rect cdt_rect;
        // cout << "count_stereo % frq_cdt: " << count_stereo % frq_cdt << endl;
        // if(config_.enabel_cdt && i % config_.frq_cdt == 0){
        if (config_.enabel_cdt)
        {
            ct_dect_src.clear();
            cdtPerception_.perception_process_bgr(croppedImg, ct_dect_src);
            // cout << "ct_dect_src size: " << ct_dect_src.size() << endl;
            cdt_rect = get_cdt_rect(ct_dect_src);
            // RCLCPP_INFO(this->get_logger(), "[global]===depth_time_cost:
            // %.2f===reciev_latency: %.2f", depth_time_cost, depth_reciev_latency);
            cout << "cdt_rect: " << cdt_rect << endl;
            cout << "cdt_rect.area: " << cdt_rect.area() << endl;
            // RCLCPP_INFO(this->get_logger(), "cdt_rect: [x=%d, y=%d, w=%d, h=%d],
            // area=%d", cdt_rect.x, cdt_rect.y, cdt_rect.width, cdt_rect.height,
            // static_cast<int>(cdt_rect.area())); ret_cdt=1;
        }
        // cv::Mat croppedImg = cv::Mat::zeros(384, 640, CV_8UC1) + 2;
        // cv::Mat lab_out;
        // std::vector<Detection> detections;
        // Mat left_copy = left.clone();
        // mulSubPerception.perception_process_bgr_no_argmax_erode(croppedImg,
        // detections, croppedImg, lab_out, config_.erode_pixel);
        mulSubPerception.perception_process_bgr_no_argmax_erode(
            croppedImg, detections, img_label, lab_out, config_.erode_pixel);
        filterLabelDect(lab_out, detections, dst_label, dect_dst);
        dst_label(cdt_rect) = -1;
        auto infer_end = chrono::high_resolution_clock::now();
        cout << "[Step 1/3] Sub-task inference done: "
            << chrono::duration<double, milli>(infer_end - infer_start).count()
            << " ms" << endl;
        //
        result.label_map = dst_label.clone();

        // 可选的点云输出
        pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud, out_xyz_rgbl_cloud;
        cv::Rect valid_depth_rect; // 有效深度区域矩形，用于在final_compared中显示

        auto fusion_start = chrono::high_resolution_clock::now();
        if (!grayImageR.empty())
        {
            // Step 2: 深度计算
            auto depth_start = chrono::high_resolution_clock::now();
            // pcl::PointCloud<pcl::PointXYZRGBL> xyz_rgbl_cloud;

            Mat disparity =
                stereo_multi_match.stereo_multi_process_depth(grayImageL, grayImageR);
            Mat depth_cal = stereo_multi_match.stereo_multi_process_filter(
                disparity, dst_label, config_.enable_height_filter_);

            // 深度图(640x480)与分割图/原图(640x384)尺寸不匹配，裁剪深度图上部384行对齐
            if (depth_cal.rows > croppedImg.rows)
            {
                depth_cal =
                    depth_cal(cv::Rect(0, 0, depth_cal.cols, croppedImg.rows)).clone();
            }

            // // 画出depth_cal有效区域的矩形边界并保存
            // valid_depth_rect = drawDepthValidRegion(depth_cal, croppedImg,
            //                                         config_.finalPicDir,
            //                                         current_image_name_);

            auto depth_end = chrono::high_resolution_clock::now();
            cout << "[Step 2/3] Depth computation done: "
                << chrono::duration<double, milli>(depth_end - depth_start).count()
                << " ms" << endl;

            // Step 3: 融合
            stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
                depth_cal, result.label_map, dect_dst, croppedImg, xyz_rgbl_cloud,
                out_xyz_rgbl_cloud);
        };

        if (config_.m_enable_debug_show)
        {
            Mat img_seg_show;
            Mat pure_seg_mat =
                drawResult(croppedImg, dst_label, dect_dst, img_seg_show);
            Mat origin_seg;
            cv::resize(pure_seg_mat, pure_seg_mat, Size(640, 384));
            cv::resize(img_seg_show, img_seg_show, Size(640, 384));

            // 在croppedImg副本上画出有效深度区域矩形框，用于final_compared可视化
            // Mat croppedImgWithRect = croppedImg.clone();
            // if (!grayImageR.empty() && valid_depth_rect.area() > 0)
            // {
            //     cv::rectangle(croppedImgWithRect, valid_depth_rect,
            //                   Scalar(0, 255, 0), 2);
            //     string rect_info =
            //         "Depth [" + to_string(valid_depth_rect.x) + "," +
            //         to_string(valid_depth_rect.y) + " " +
            //         to_string(valid_depth_rect.width) + "x" +
            //         to_string(valid_depth_rect.height) + "]";
            //     cv::putText(croppedImgWithRect, rect_info,
            //                 Point(valid_depth_rect.x,
            //                       max(valid_depth_rect.y - 10, 15)),
            //                 FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            // }

            // cv::hconcat(croppedImgWithRect, pure_seg_mat,
            cv::hconcat(croppedImg, pure_seg_mat,
                        origin_seg); // 拼接带depth矩形的原图和分割图
            cv::hconcat(origin_seg, img_seg_show, origin_seg);
            Mat xyz_rgbl, final_compared;
            stereo_point_cloud stereoPointCloud;
            if (!grayImageR.empty())
            {
                stereoPointCloud.show_xyz_rgbl_plane_point_cloud_final(
                    out_xyz_rgbl_cloud, xyz_rgbl);
                cv::vconcat(origin_seg, xyz_rgbl,
                            final_compared); // 再将图片3拼接到结果中
                string finalPcdPath =
                    config_.finalPcdDir + current_image_name_ + "_sub_cdt";
                savePcdfile_with_rgb_label(out_xyz_rgbl_cloud, finalPcdPath);
            }
            else
            {
                final_compared = origin_seg;
            }

            // 构造输出路径: finalPicDir + 原文件名 + "_cdt.jpg"
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
int main(int argc, char** argv)
{
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
    // "/home/youfeng/debug/02/26/rosbag_LK-MR2P1US000017_navigation_202602262006/stereo_output_rosbag_LK-MR2P1US000017_navigation_202602262006_0/images/";
    // string input_dir =
    // "/home/youfeng/debug/02/26/rosbag_LK-MR2P1US000017_navigation_202602262006/stereo_output_rosbag_LK-MR2P1US000017_navigation_202602262006_0/images/";
    // string input_dir =
    // "/home/youfeng/debug/02/27/rosbag_MR1P1251US0008152_navigation_202602271431/stereo_output_rosbag_MR1P1251US0008152_navigation_202602271431_0/images/extracted_interval/";
    // string input_dir = "/home/youfeng/debug/03/03/allways_avioding/rosbag_record/rosbag_LK-MR2P1US000015_camera_202603031111/stereo_output_rosbag_LK-MR2P1US000015_camera_202603031111_0/images/extracted_interval/";
    // string input_dir = "/home/youfeng/debug/custom/6506/0306/stereo/pic/";
    // string input_dir = "/home/youfeng/debug/03/03/userdata/rosbag_record/rosbag_LK-MR6P1US000107_camera_202603030959/stereo_output_rosbag_LK-MR6P1US000107_camera_202603030959_0/images/extracted_interval/brick/";
    string input_dir = "/home/youfeng/debug/03/06/bug/userdata/rosbag_record/rosbag_LK-MR6P1US000107_navigation_202603061404/stereo_output_rosbag_LK-MR6P1US000107_navigation_202603061404_0_filtered_20260306_1405_to_20260306_1408/images/no_point/";
    cout << "\nInput directory: " << input_dir << endl;

    // 构造最终结果输出目录
    const string mode_suffix =
        to_string(config.infer_mode) + "_" + to_string(config.erode_pixel);
    if (config.infer_mode == 5)
    {
        config.finalPicDir = input_dir + "/cdt_mul_" + mode_suffix + "_0306/";
    }
    else if (config.infer_mode == 6)
    {
        config.finalPicDir =
            input_dir + "/cdt_sub_" + mode_suffix + "_0306/";
    }
    config.finalPcdDir = input_dir + "/pcd_" + mode_suffix +
        "_0306/"; // 设置最终PCD输出目录

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
        for (const auto& entry : fs::directory_iterator(input_dir))
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
    catch (const fs::filesystem_error& e)
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
        const string& image_path = image_files[i];

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
        Mat right_img; // 默认没有右图

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

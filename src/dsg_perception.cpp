/**************************************************************
 *  Copyright (c):   YJ Intelligent Technology Co., LTD.
 *  Department:  Perception
 *  Description: dsg_perception class
 *
 *  @author:     YouFeng
 *  @data        2026/04/01
 **************************************************************/
#include <iostream>
#include <cmath>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "perception.h"
#include "dsg_perception.h"

using namespace std;
using namespace cv;

// ============== 检测类别名称 ==============
std::vector<std::string> dsg_class_names = {
    "pole", "obstacle_tree", "fixed_obstacle", "cars",
    "static_obstacle", "dynamic_obstacle", "charge_station", "person"
};

// ============== YOLOv5 解码相关常量 ==============
const int DSG_STRIDES[] = {8, 16, 32};

const int DSG_FEAT_H[] = {48, 24, 12};  // 384/8=48, 384/16=24, 384/32=12
const int DSG_FEAT_W[] = {80, 40, 20};  // 640/8=80, 640/16=40, 640/32=20

const int DSG_NUM_ANCHORS_PER_SCALE[] = {
    DSG_FEAT_H[0] * DSG_FEAT_W[0] * 3,  // 48*80*3 = 11520
    DSG_FEAT_H[1] * DSG_FEAT_W[1] * 3,  // 24*40*3 = 2880
    DSG_FEAT_H[2] * DSG_FEAT_W[2] * 3   // 12*20*3 = 720
};

const float DSG_ANCHORS[3][3][2] = {
    {{10, 13}, {16, 30}, {33, 23}},      // stride 8
    {{30, 61}, {62, 45}, {59, 119}},     // stride 16
    {{116, 90}, {156, 198}, {373, 326}}  // stride 32
};

// ============== 坐标解码 ==============
static void decode_box(int box_idx,
                       float sigmoid_x, float sigmoid_y,
                       float sigmoid_w, float sigmoid_h,
                       float &center_x, float &center_y,
                       float &width, float &height)
{
    int scale = 0;
    int idx_in_scale = box_idx;

    if (box_idx < DSG_NUM_ANCHORS_PER_SCALE[0]) {
        scale = 0;
    } else if (box_idx < DSG_NUM_ANCHORS_PER_SCALE[0] + DSG_NUM_ANCHORS_PER_SCALE[1]) {
        scale = 1;
        idx_in_scale = box_idx - DSG_NUM_ANCHORS_PER_SCALE[0];
    } else {
        scale = 2;
        idx_in_scale = box_idx - DSG_NUM_ANCHORS_PER_SCALE[0] - DSG_NUM_ANCHORS_PER_SCALE[1];
    }

    int feat_h = DSG_FEAT_H[scale];
    int feat_w = DSG_FEAT_W[scale];
    int stride = DSG_STRIDES[scale];

    int anchor_idx  = idx_in_scale / (feat_h * feat_w);
    int spatial_idx = idx_in_scale % (feat_h * feat_w);
    int grid_y = spatial_idx / feat_w;
    int grid_x = spatial_idx % feat_w;

    float anchor_w = DSG_ANCHORS[scale][anchor_idx][0];
    float anchor_h = DSG_ANCHORS[scale][anchor_idx][1];

    center_x = (sigmoid_x * 2.0f - 0.5f + grid_x) * stride;
    center_y = (sigmoid_y * 2.0f - 0.5f + grid_y) * stride;
    width    = std::pow(sigmoid_w * 2.0f, 2) * anchor_w;
    height   = std::pow(sigmoid_h * 2.0f, 2) * anchor_h;
}

// ============== IoU ==============
static float intersection_over_union(const Bbox &bbox1, const Bbox &bbox2)
{
    float x1 = std::max(bbox1.xmin, bbox2.xmin);
    float y1 = std::max(bbox1.ymin, bbox2.ymin);
    float x2 = std::min(bbox1.xmax, bbox2.xmax);
    float y2 = std::min(bbox1.ymax, bbox2.ymax);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin);
    float area2 = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin);

    return inter / (area1 + area2 - inter);
}

// ============== NMS ==============
static std::vector<Detection> non_max_suppression(
    const std::vector<Detection> &detections,
    float iou_threshold, int max_det)
{
    std::vector<Detection> result;
    std::vector<Detection> det_copy = detections;
    std::sort(det_copy.begin(), det_copy.end(),
              [](const Detection &a, const Detection &b) { return a.score > b.score; });

    std::vector<bool> keep(det_copy.size(), true);
    for (size_t i = 0; i < det_copy.size(); ++i) {
        if (!keep[i]) continue;
        result.push_back(det_copy[i]);
        if ((int)result.size() >= max_det) break;
        for (size_t j = i + 1; j < det_copy.size(); ++j) {
            if (keep[j] &&
                intersection_over_union(det_copy[i].bbox, det_copy[j].bbox) > iou_threshold)
                keep[j] = false;
        }
    }
    return result;
}

// ============================================================
//  perception_init  （参考 multi_perception 直接使用类成员）
// ============================================================
void dsg_perception::perception_init(const char *model_file_name)
{
    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    cout << "DNN runtime version: " << hbDNNGetVersion() << endl;

    int inCount_ret  = hbDNNGetInputCount(&input_count, dnn_handle);
    int outCount_ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    if (inCount_ret || outCount_ret != 0)
        cout << "hbDNNGetInputCount/hbDNNGetOutputCount failed" << endl;

    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data());
}

// ============================================================
//  prepare_tensor  （不再重复调用 count API，参考 multi_perception）
// ============================================================
int dsg_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor)
{
    this->input = input_tensor;
    for (int i = 0; i < input_count; i++) {
        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);

        model_height = input[i].properties.validShape.dimensionSize[2];
        model_width  = input[i].properties.validShape.dimensionSize[3];
        cout << "model_height/model_width: " << model_height << "/" << model_width << endl;

        int input_memSize         = input[i].properties.alignedByteSize;
        int input_allocCached_ret = hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
        input[i].properties.alignedShape = input[i].properties.validShape;

        if (inTensor_ret || input_allocCached_ret != 0)
            cout << "hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed" << endl;
    }

    this->output = output_tensor;
    for (int i = 0; i < output_count; i++) {
        int outTensor_ret         = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int output_memSize        = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);
        if (outTensor_ret || output_allocCached_ret != 0)
            cout << "hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed" << endl;
        int *s = output[i].properties.validShape.dimensionSize;
        cout << "output[" << i << "] shape: [" << s[0] << "," << s[1] << "," << s[2] << "," << s[3]
             << "] alignedByteSize=" << output_memSize << endl;
    }
    return 0;
}

// ============================================================
//  read_image_2_tensor_as_nv12  （输入已裁剪为模型尺寸）
// ============================================================
int dsg_perception::read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor)
{
    hbDNNTensor *inp = input_tensor;
    hbDNNTensorProperties Properties = inp->properties;

    int input_h = Properties.validShape.dimensionSize[2];
    int input_w = Properties.validShape.dimensionSize[3];

    if (input_h % 2 || input_w % 2) {
        cout << "Input img height and width must be aligned by 2!" << endl;
        return -1;
    }

    cv::Mat yuv_mat;
    cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

    auto data = inp->sysMem[0].virAddr;
    int32_t y_size = input_h * input_w;
    memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

    int32_t uv_height = input_h / 2;
    int32_t uv_width  = input_w / 2;
    uint8_t *nv12   = reinterpret_cast<uint8_t *>(data) + y_size;
    uint8_t *u_data = nv12_data + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        if (u_data && v_data) {
            *nv12++ = *u_data++;
            *nv12++ = *v_data++;
        }
    }
    return 0;
}

// ============================================================
//  perception_preprocess_bgr  （预处理 + 运行推理）
//  参考 new_x5_night/main.cc 的推理流程
// ============================================================
void dsg_perception::perception_preprocess_bgr(Mat &mat)
{
    // 如果输入尺寸与模型不符则先 resize（参考代码对齐保护）
    Mat input_mat;
    if (mat.rows != model_height || mat.cols != model_width) {
        cv::resize(mat, input_mat, cv::Size(model_width, model_height));
    } else {
        input_mat = mat;
    }

    int nv12Ret = read_image_2_tensor_as_nv12(input_mat, input_tensors.data());
    if (nv12Ret != 0) cout << "read_image_2_tensor_as_nv12 failed" << endl;

    for (int j = 0; j < input_count; j++)
        hbSysFlushMem(&input_tensors[j].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // 参考代码：output 指针由 hbDNNInfer 管理，推理后可能被更新
    output = output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    infer_ctrl_param.bpuCoreId = 0;

    int infer_ret = hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handle, &infer_ctrl_param);
    if (infer_ret != 0) cout << "hbDNNInfer failed" << endl;

    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    if (task_ret != 0) cout << "hbDNNWaitTaskDone failed" << endl;
}

// ============================================================
//  get_detect_result_no_argmax
//  output[0]: YOLO 检测（float, shape [1, num_boxes, num_features]）
//  output[1]: DSG 分割（参考 new_x5_night/main.cc）
//    - num_classes==1: int32_t argmax 已在模型内完成
//    - num_classes>1 : float 多通道，代码侧手动 argmax（NCHW）
//  DSG 标签映射: raw 0→1(背景), 1→5(静态障碍), 2→3(路面)
// ============================================================
Mat dsg_perception::get_detect_result_no_argmax(
    hbDNNTensor *output, float cls_thre, float iou_thre,
    std::vector<Detection> &detections, Mat &img_label)
{
    // ---------- 检测解码 ----------
    auto data_detect = reinterpret_cast<float *>(output[0].sysMem[0].virAddr);
    int *shape       = output[0].properties.validShape.dimensionSize;
    int num_boxes    = shape[1];
    int num_features = shape[2];

    float scale_x = ori_width  / float(model_width);
    float scale_y = ori_height / float(model_height);

    for (int i = 0; i < num_boxes; i++) {
        float confidence = data_detect[i * num_features + 4];
        if (confidence <= cls_thre) continue;

        float sigmoid_x = data_detect[i * num_features + 0];
        float sigmoid_y = data_detect[i * num_features + 1];
        float sigmoid_w = data_detect[i * num_features + 2];
        float sigmoid_h = data_detect[i * num_features + 3];

        float cx, cy, w, h;
        decode_box(i, sigmoid_x, sigmoid_y, sigmoid_w, sigmoid_h, cx, cy, w, h);

        float xmin = (cx - w / 2) * scale_x;
        float ymin = (cy - h / 2) * scale_y;
        float xmax = (cx + w / 2) * scale_x;
        float ymax = (cy + h / 2) * scale_y;

        xmin = xmin > 0 ? xmin : 0;
        ymin = ymin > 0 ? ymin : 0;
        xmax = xmax < ori_width  ? xmax : ori_width;
        ymax = ymax < ori_height ? ymax : ori_height;

        int class_id = 0;
        float max_prob = 0.0f;
        for (int c = 0; c < num_features - 5; c++) {
            float p = data_detect[i * num_features + 5 + c];
            if (p > max_prob) { max_prob = p; class_id = c; }
        }

        Bbox bbox(xmin, ymin, xmax, ymax);
        detections.push_back(Detection(class_id, confidence, bbox,
                                       dsg_class_names[class_id].c_str()));
    }

    detections = non_max_suppression(detections, iou_thre, 40);

    // ---------- 分割解码（参考 new_x5_night/main.cc，DSG 标签映射：0→1, 1→5, 2→3）----------
    int *seg_shape   = output[1].properties.validShape.dimensionSize;
    int num_classes  = seg_shape[1];   // 1=argmax int32; >1=float 多通道
    int seg_height   = seg_shape[2];
    int seg_width    = seg_shape[3];

    // DSG 原始类别 → 应用层 label
    // raw: 0=unlabel→1(back), 1=background→1(back), 2=obstacle→5(stat), 3=passable→3(road)
    static const uint8_t dsg_label_map[] = {1, 1, 5, 3};
    auto map_label = [](int raw) -> uint8_t {
        if (raw >= 0 && raw < 4) return dsg_label_map[raw];
        return static_cast<uint8_t>(raw);
    };

    if (num_classes == 1) {
        // 模型内已完成 argmax，输出 int32_t 标签图
        auto data_seg = reinterpret_cast<int32_t *>(output[1].sysMem[0].virAddr);
        for (int h = 0; h < seg_height; h++) {
            for (int w = 0; w < seg_width; w++) {
                int raw = static_cast<int>(data_seg[h * seg_width + w]);
                int orig_x = static_cast<int>(w * scale_x);
                int orig_y = static_cast<int>(h * scale_y);
                if (orig_y < img_label.rows && orig_x < img_label.cols)
                    img_label.at<uint8_t>(orig_y, orig_x) = map_label(raw);
            }
        }
    } else {
        // 模型输出 float 多通道概率，NCHW 布局，代码侧 argmax
        auto data_seg = reinterpret_cast<float *>(output[1].sysMem[0].virAddr);
        for (int h = 0; h < seg_height; h++) {
            for (int w = 0; w < seg_width; w++) {
                int max_class = 0;
                float max_score = data_seg[0 * seg_height * seg_width + h * seg_width + w];
                for (int c = 1; c < num_classes; c++) {
                    float score = data_seg[c * seg_height * seg_width + h * seg_width + w];
                    if (score > max_score) { max_score = score; max_class = c; }
                }
                int orig_x = static_cast<int>(w * scale_x);
                int orig_y = static_cast<int>(h * scale_y);
                if (orig_y < img_label.rows && orig_x < img_label.cols)
                    img_label.at<uint8_t>(orig_y, orig_x) = map_label(max_class);
            }
        }
    }

    return img_label;
}

// ============================================================
//  perception_postprocess_no_argmax
//  参考 new_x5_night/main.cc：推理后立即 flush 全部输出张量
// ============================================================
void dsg_perception::perception_postprocess_no_argmax(
    std::vector<Detection> &detections, Mat &img_label)
{
    // 用推理后的 output 指针 flush（hbDNNInfer 可能更新了该指针）
    for (int i = 0; i < output_count; i++)
        hbSysFlushMem(&(output[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    get_detect_result_no_argmax(output, 0.45f, 0.45f, detections, img_label);
}

// ============================================================
//  false_positive_suppress3  （膨胀 class2 / class3，无边界平滑）
// ============================================================
int dsg_perception::false_positive_suppress3(cv::Mat &lab_ori, cv::Mat &lab_dst, int threshold)
{
    lab_dst = lab_ori.clone();
    if (threshold == 0) return 0;

    const int rows = lab_ori.rows;
    const int cols = lab_ori.cols;

    cv::Mat mask_2, mask_3;
    cv::Mat mat_2 = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat mat_3 = cv::Mat::zeros(rows, cols, CV_8UC1);

    cv::compare(lab_ori, 2, mask_2, cv::CMP_EQ);
    lab_ori.copyTo(mat_2, mask_2);

    cv::compare(lab_ori, 3, mask_3, cv::CMP_EQ);
    lab_ori.copyTo(mat_3, mask_3);

    cv::Mat background = lab_ori.clone();
    background.setTo(1, mask_2);
    background.setTo(1, mask_3);
    lab_dst = background.clone();

    if (threshold > 2 && threshold < 100) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(threshold, threshold));
        cv::dilate(mat_2, mat_2, kernel);
        lab_dst.setTo(2, mat_2);
    } else if (threshold >= 100 && threshold < 200) {
        int road_threshold = threshold - 100;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(road_threshold, road_threshold));
        cv::dilate(mat_3, mat_3, kernel);
        lab_dst.setTo(3, mat_3);
    } else if (threshold >= 200 && threshold < 300) {
        int ther = threshold - 200;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ther, ther));
        cv::dilate(mat_2, mat_2, kernel);
        cv::dilate(mat_3, mat_3, kernel);
        lab_dst.setTo(2, mat_2);
        lab_dst.setTo(3, mat_3);
    }

    return 0;
}

// ============================================================
//  false_positive_suppress4  （膨胀 class2 / class3，带边界平滑）
// ============================================================
int dsg_perception::false_positive_suppress4(cv::Mat &lab_ori, cv::Mat &lab_dst, int threshold)
{
    lab_dst = lab_ori.clone();
    if (threshold == 0) return 0;

    const int rows = lab_ori.rows;
    const int cols = lab_ori.cols;

    cv::Mat mask_2, mask_3;
    cv::Mat mat_2 = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat mat_3 = cv::Mat::zeros(rows, cols, CV_8UC1);

    cv::compare(lab_ori, 2, mask_2, cv::CMP_EQ);
    lab_ori.copyTo(mat_2, mask_2);

    cv::compare(lab_ori, 3, mask_3, cv::CMP_EQ);
    lab_ori.copyTo(mat_3, mask_3);

    cv::Mat background = lab_ori.clone();
    background.setTo(1, mask_2);
    background.setTo(1, mask_3);
    lab_dst = background.clone();

    auto smooth_mat2 = [&]() {
        cv::Mat smooth_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mat_2, mat_2, cv::MORPH_CLOSE, smooth_kernel);
        cv::Mat boundary_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mat_2, mat_2, boundary_kernel);
        cv::dilate(mat_2, mat_2, boundary_kernel);
    };

    if (threshold > 2 && threshold < 100) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(threshold, threshold));
        cv::dilate(mat_2, mat_2, kernel);
        smooth_mat2();
        lab_dst.setTo(2, mat_2);
    } else if (threshold >= 100 && threshold < 200) {
        int road_threshold = threshold - 100;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(road_threshold, road_threshold));
        cv::dilate(mat_3, mat_3, kernel);
        lab_dst.setTo(3, mat_3);
    } else if (threshold >= 200 && threshold < 300) {
        int ther = threshold - 200;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ther, ther));
        cv::dilate(mat_2, mat_2, kernel);
        cv::dilate(mat_3, mat_3, kernel);
        smooth_mat2();
        lab_dst.setTo(2, mat_2);
        lab_dst.setTo(3, mat_3);
    }

    return 0;
}

// ============================================================
//  perception_process_bgr_no_argmax_erode  （完整流水线）
// ============================================================
void dsg_perception::perception_process_bgr_no_argmax_erode(
    Mat &bgr, std::vector<Detection> &detections,
    Mat &img_label, Mat &lab_dst, int m_erode_pixel)
{
    perception_preprocess_bgr(bgr);
    perception_postprocess_no_argmax(detections, img_label);
    false_positive_suppress4(img_label, lab_dst, m_erode_pixel);
    task_release();
}

// ============================================================
//  以下为兼容旧接口（DSG 单分割推理）
// ============================================================

int dsg_perception::BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12)
{
    auto height = bgr_mat.rows;
    auto width  = bgr_mat.cols;
    if (height % 2 || width % 2) {
        std::cerr << "input img height and width must aligned by 2!" << std::endl;
        return -1;
    }
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    if (yuv_mat.data == nullptr) {
        std::cerr << "yuv_mat.data is null pointer" << std::endl;
        return -1;
    }

    auto *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12  = cv::Mat(height * 3 / 2, width, CV_8UC1);
    auto *ynv12 = img_nv12.ptr<uint8_t>();

    int32_t uv_height = height / 2;
    int32_t uv_width  = width  / 2;
    int32_t y_size    = height * width;
    memcpy(ynv12, yuv, y_size);

    uint8_t *nv12   = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return 0;
}

int dsg_perception::prepare_mat_nv12(Mat originMat)
{
    hbDNNTensor *inp = input_tensors.data();
    hbDNNTensorProperties Properties = inp->properties;
    int input_h = Properties.validShape.dimensionSize[1];
    int input_w = Properties.validShape.dimensionSize[2];
    if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        input_h = Properties.validShape.dimensionSize[2];
        input_w = Properties.validShape.dimensionSize[3];
    }

    cv::Mat mat;
    mat.create(input_h, input_w, originMat.type());
    cv::resize(originMat, mat, mat.size(), 0, 0);

    if (input_h % 2 || input_w % 2) {
        cout << "input img height and width must aligned by 2!" << endl;
        return -1;
    }
    cv::Mat yuv_mat;
    cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

    auto data = inp->sysMem[0].virAddr;
    int32_t y_size    = input_h * input_w;
    memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

    int32_t uv_height = input_h / 2;
    int32_t uv_width  = input_w  / 2;
    uint8_t *nv12   = reinterpret_cast<uint8_t *>(data) + y_size;
    uint8_t *u_data = nv12_data + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return 0;
}

void dsg_perception::perception_process(Mat &mat)
{
    int nv12Ret = prepare_mat_nv12(mat);
    for (int j = 0; j < input_count; j++)
        hbSysFlushMem(&input_tensors[j].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    this->output = output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    infer_ctrl_param.bpuCoreId = 0;

    int infer_ret = hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handle, &infer_ctrl_param);
    if (infer_ret != 0) cout << "hbDNNInfer failed" << endl;
    int task_ret  = hbDNNWaitTaskDone(task_handle, 0);
    if (task_ret  != 0) cout << "hbDNNWaitTaskDone failed" << endl;
}

int dsg_perception::get_tensor_hwc_index(hbDNNTensor *tensor, int *h_index, int *w_index, int *c_index)
{
    if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        *h_index = 1; *w_index = 2; *c_index = 3;
    } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW ||
               tensor->properties.tensorLayout == HB_DNN_LAYOUT_NONE) {
        *c_index = 1; *h_index = 2; *w_index = 3;
    } else {
        return -1;
    }
    return 0;
}

void dsg_perception::lab_match(hbDNNTensor *output_tensors, Mat &lab_out)
{
    hbDNNTensor *tensors = output_tensors;
    int h_index, w_index, c_index;
    get_tensor_hwc_index(&tensors[0], &h_index, &w_index, &c_index);
    int height = tensors[0].properties.validShape.dimensionSize[h_index];
    int width  = tensors[0].properties.validShape.dimensionSize[w_index];

    uint8_t *result_ptr = lab_out.ptr<uint8_t>();
    int8_t  *data       = reinterpret_cast<int8_t *>(tensors[0].sysMem[0].virAddr);

    if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int8_t *top_index = data + (h * width + w) * 4;
                uint8_t match_label;
                if      (top_index[0] == 0) match_label = 1;
                else if (top_index[0] == 1) match_label = 5;
                else if (top_index[0] == 2) match_label = 3;
                else                        match_label = top_index[0];
                result_ptr[h * width + w] = match_label;
            }
        }
    }
}

void dsg_perception::perception_postprocess_match(Mat &lab_out)
{
    // 多任务模型: output[0]=检测, output[1]=分割; 冲刷分割 tensor
    int seg_idx = (output_count > 1) ? 1 : 0;
    hbSysFlushMem(&(output_tensors[seg_idx].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    lab_match(output_tensors.data() + seg_idx, lab_out);
}

void dsg_perception::process_infer_match(Mat &mat, Mat &lab_out)
{
    perception_process(mat);
    perception_postprocess_match(lab_out);
    task_release();
}

// ============================================================
//  资源释放
// ============================================================
void dsg_perception::task_release()
{
    int res_ret = hbDNNReleaseTask(task_handle);
    if (res_ret != 0) cout << "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void dsg_perception::perception_release()
{
    for (int i = 0; i < input_count; i++) {
        int infree_ret = hbSysFreeMem(&(input_tensors[i].sysMem[0]));
        if (infree_ret != 0) cout << "hbSysFreeMem failed" << endl;
    }
    for (int i = 0; i < output_count; i++) {
        int outfree_ret = hbSysFreeMem(&(output_tensors[i].sysMem[0]));
        if (outfree_ret != 0) cout << "hbSysFreeMem failed" << endl;
    }
    int dnn_ret = hbDNNRelease(packed_dnn_handle);
    if (dnn_ret != 0) cout << "hbDNNRelease failed" << endl;
}

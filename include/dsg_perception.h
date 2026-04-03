/**************************************************************
 *  Copyright (c):   YJ Intelligent Technology Co., LTD.
 *  Department:  Perception
 *  Description: dsg_perception class
 *
 *  @author:     YouFeng
 *  @data        2026/01/15 14:58
 **************************************************************/
#include <iostream>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "perception.h"


using namespace std;
using namespace cv;

class dsg_perception {
private:
    int num_classes;
    string model_name;
    int input_count = 0;
    int output_count = 0;
    hbDNNTensor *input;
    hbDNNTensor *output;

    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNTaskHandle_t task_handle = nullptr;
    hbPackedDNNHandle_t packed_dnn_handle = nullptr;

public:
    int model_height;
    int model_width;

    int ori_height;
    int ori_width;

    // ========== 初始化与资源管理 ==========
    void perception_init(const char *model_file_name);
    int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    void task_release();
    void perception_release();

    // ========== 预处理与推理 ==========
    int read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor);
    void perception_preprocess_bgr(Mat &mat);

    // ========== 后处理：检测 + 分割 ==========
    void perception_postprocess_no_argmax(std::vector<Detection> &detections, Mat &img_label);
    Mat get_detect_result_no_argmax(hbDNNTensor *output, float cls_thre, float iou_thre,
                                    std::vector<Detection> &detections, Mat &img_label);

    // ========== 形态学假阳性抑制 ==========
    int false_positive_suppress3(cv::Mat &lab_ori, cv::Mat &lab_dst, int threshold);
    int false_positive_suppress4(cv::Mat &lab_ori, cv::Mat &lab_dst, int threshold);

    // ========== 完整推理流水线 ==========
    void perception_process_bgr_no_argmax_erode(Mat &bgr, std::vector<Detection> &detections,
                                                Mat &img_label, Mat &lab_dst, int m_erode_pixel);

    // ========== DSG 单分割推理（兼容旧接口）==========
    int BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12);
    int prepare_mat_nv12(Mat originMat);
    void perception_process(Mat &mat);
    int get_tensor_hwc_index(hbDNNTensor *tensor, int *h_index, int *w_index, int *c_index);
    void lab_match(hbDNNTensor *output_tensors, Mat &lab_out);
    void perception_postprocess_match(Mat &lab_out);
    void process_infer_match(Mat &mat, Mat &lab_out);
};

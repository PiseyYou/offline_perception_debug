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
#include "perception_common.h"


using namespace std;
using namespace cv;

class dsg_perception {
private:
    int num_classes;
    string model_name;
    int input_count;
    int output_count;
    hbDNNTensor *input;
    hbDNNTensor *output;

    int height;
    int width;

    int yuv_height;
    int yuv_width;

    cv::Point mp;
    std::vector<cv::Point> max_contours;

    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    hbDNNHandle_t dnn_handle=nullptr;
    hbDNNTaskHandle_t task_handle=nullptr;
    hbPackedDNNHandle_t packed_dnn_handle=nullptr;

public:
    int BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12);
    void perception_init(const char *model_file_name);
    int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);

    int prepare_mat_nv12(Mat originMat);
    void perception_process(Mat &mat);
    // void perception_postprocess(Mat &mat, int count);
    // cv::Mat perception_postprocess();
    // cv::Mat process_infer(Mat &mat);

    // Simplified functions without Perception struct
    // Mat argmax_and_draw(cv::Mat &input_mat, int count);
    // cv::Mat do_argmax(hbDNNTensor *output_tensors);
    int get_tensor_hwc_index(hbDNNTensor *tensor, int *h_index, int *w_index, int *c_index);


    void lab_match(hbDNNTensor *output_tensors, Mat& lab_out);
    void perception_postprocess_match(Mat& lab_out);
    void process_infer_match(Mat &mat, Mat& lab_out);

    void task_release();
    void perception_release();
};
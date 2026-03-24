/**************************************************************
 *  Copyright (c):   LF Intelligent Technology Co., LTD.
 *  Department:  Perception
 *  Description: TODO
 *
 *  @author:     YouFeng
 *  @data        2024/09/29 下午16:44
 **************************************************************/

#include <iostream>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <opencv2/opencv.hpp>

// #include "gflags/gflags.h"
#include "perception_common.h"

#include <algorithm>
#include <vector>
#include <memory>

using namespace std;
using namespace cv;

#define EMPTY ""
#define NMS_MAX_INPUT (400)
#define DETECTION_CLASS_NUM 9

#define NUM_CLASS_NANO 9
#define NUM_PARAM 41

struct DetConfig {
    std::vector<int> strides;
    int class_num;
    std::vector<std::string> class_names;
};


struct ScoreId {
    float score;
    int id;
};

class det_perception {
protected:
    float iou_threshold = 0.5f;
    float scale=0.8;

    int num_classes;
    string model_name;
    int input_count;
    int output_count;
    hbDNNTensor *input;
    hbDNNTensor *output;

    int topk = 40;
    cv::Mat after_sizing; // this is image after centerpad/resize
    int orig_h; // original size of the input image
    int orig_w;
    bool bgr = true;
    DetConfig det_config_;

    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    hbDNNHandle_t dnn_handle=nullptr;
    hbDNNTaskHandle_t task_handle=nullptr;
    hbPackedDNNHandle_t packed_dnn_handle=nullptr;

//    std::vector<unsigned char> det_compressed_data;
//    std::vector<int> det_compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};

//    sensor_msgs::msg::CompressedImage det_compressed_msg;
//    det_compressed_msg.header = left->header;  // 复制原消息的header
//    det_compressed_msg.format = "jpeg";  // 设置格式为jpeg

public:
    int fov_w;
    int fov_h;
    int model_height;
    int model_width;

//    map<int, string> det_map_class;
//    int dect_class_num;

    int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    int read_image_2_tensor_as_rgb(Mat &bgr_mat);
    int read_nv12(Mat &yuv_mat, hbDNNTensor *input_tensor);
    int read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor);
    void perception_init(const char *model_file_name, float dect_threshold);

    void check_dist(float &area, float &min_dist, float &max_dist);
    void det_side_persion_info(std::vector<Detection> &dect_src, int &person_num,
                               float &person_max_score, float &person_max_area, float &person_min_dist, float &person_max_dist);

    float det_sigmoid(const float input);
    float softmax8(const float *input);

    vector<Detection> decode_nanodet(hbDNNTensor *output_tensors);
    float m_conf_threshold;
    void decode_nanodet_nhwc(
            // Input network output
            const float * outputs,
            // Inputs parameter
            const int stride,
            const int h,
            const int w,
            const float conf_threshold,
            // Outputs
            std::vector<Detection> &dets);

    void nms(std::vector<Detection> &input, float iou_threshold, int top_k, std::vector<Detection> &result, bool suppress);

    void perception_process(Mat &mat);
    vector<Detection> refine_rect(vector<Detection> det, vector<Detection> &picDet);
    void perception_postprocess_nanodet(Mat mat, vector<Detection> &picDet);

    void perception_process_nv12(Mat &yuv_mat);
    vector<Detection> refine_rect_nv12(std::vector<Detection> det, vector<Detection> &picDet);
    void perception_postprocess_nanodet_nv12(Mat yuv_mat, vector<Detection> &picDet);
    bool find_person_area(std::vector<Detection> dect_src, int det_person_area);

    void task_release();
    void perception_release();

};

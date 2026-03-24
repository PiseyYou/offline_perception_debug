#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <utility>
#include <regex>

#include "dnn/hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "perception_common.h"

using namespace std;
using namespace cv;


class multi_perception {
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

    int ori_height=384;
    int ori_width=640;

    void perception_init(const char *model_file_name);
    void prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    int read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor);
    void perception_preprocess_bgr(Mat &mat);

    void perception_postprocess_no_argmax(std::vector<Detection> &detections, Mat &img_label);
    Mat get_detect_result_no_argmax(hbDNNTensor *output, float cls_thre, float iou_thre, std::vector<Detection> &detections, Mat &img_label);

    int false_positive_suppress3(cv::Mat& lab_ori, cv::Mat& lab_dst, int threshold);
    int false_positive_suppress4(cv::Mat& lab_ori, cv::Mat& lab_dst, int threshold);
    void perception_process_bgr_no_argmax_erode(Mat &bgr, std::vector<Detection> &detections, Mat &img_label, Mat &lab_dst, int m_erode_pixel);

    void perception_postprocess_no_argmax_mul(std::vector<Detection> &detections, Mat &img_label);
    Mat get_detect_result_no_argmax_mul(hbDNNTensor *output, float cls_thre, float iou_thre, std::vector<Detection> &detections, cv::Mat &img_label);
    void perception_process_bgr_no_argmax_erode_mul(Mat &bgr, std::vector<Detection> &detections, Mat &img_label, int m_erode_pixel);
    void task_release();
    void perception_release();
};
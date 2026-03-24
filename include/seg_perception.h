#include <iostream>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include "perception_common.h"

using namespace std;
using namespace cv;

class seg_perception {
private:
    int num_classes;
    string model_name;
    int input_count;
    int output_count;
    hbDNNTensor *input;
    hbDNNTensor *output;

    int model_width;
    int model_height;

    int yuv_height;
    int yuv_width;

    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    hbDNNHandle_t dnn_handle=nullptr;
    hbDNNTaskHandle_t task_handle=nullptr;
    hbPackedDNNHandle_t packed_dnn_handle=nullptr;

    cv::Mat m_close_element;
    int m_min_threshold;
    float seg_ignore_ratio;

    cv::Point mp;
    std::vector<cv::Point> max_contours;

    int model_count = 0;
    const char **model_name_list;

public:

    void perception_init(const char *model_file_name);
    int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    int prepare_mat_nv12(Mat originMat);

    void perception_process(Mat &mat);
    Mat perception_postprocess_int64();

    string check_lab(Mat& lab_seg);
    int false_positive_suppress3(cv::Mat& lab_ori, cv::Mat& lab_dst, int thershold);

    bool find_max_person_area(Mat &seg_lab, int area);
    bool check_person(Mat& lab_seg);
    Mat perception_postprocess_int64_erode(Mat &img, int erode_pixel);

    void task_release();
    void perception_release();
};

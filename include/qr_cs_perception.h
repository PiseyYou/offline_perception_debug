#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <utility>
#include <regex>

#include "dnn/hb_dnn.h"
// #include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "perception_common.h"
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

class qr_cs_perception
{
private:
    int num_classes = 1;
    string model_name;
    int input_count;
    int output_count;

    int model_count = 0;
    const char **model_name_list;

    hbDNNTensor *input;
    hbDNNTensor *output;

    int height = 384;
    int width = 640;

    //    int yuv_height;
    //    int yuv_width;

    std::vector<cv::Point> max_contours;
    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNTaskHandle_t task_handle = nullptr;
    hbPackedDNNHandle_t packed_dnn_handle = nullptr;

    float nms_threshold  = 0.7;
    float score_threshold = 0.25;
    int reg = 16;
    int mces = 32;
    // 四个level特征图height尺寸
    int32_t H_4 = 96;
    int32_t H_8 = 48;
    int32_t H_16 = 24;
    int32_t H_32 = 12;
    // 四个level特征图width尺寸
    int32_t W_4 = 160;
    int32_t W_8 = 80;
    int32_t W_16 = 40;
    int32_t W_32 = 20;
    // 输出映射顺序
    int order[10] = {1, 0, 2, 4, 3, 5, 7, 6, 8, 9};
    // 存储解码信息
    std::vector<cv::Rect2d> decoded_bboxes_all_; // 解码出的边界框
    std::vector<float> decoded_scores_all_;      // 储存解码出得分
    // std::vector<int> decoded_classes_all_;              // 储存解码出的类别
    std::vector<std::vector<float>> decoded_mces_all_; // 每个框的掩码系数

public:
    cv::Point mp;
    void perception_init(const char *model_file_name);
    int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    int prepare_mat_nv12(Mat originMat);
    void perception_process(Mat &mat);
    Mat perception_postprocess_int64();
    Mat perception_postprocess_yolov8n_int64();
    void ProcessFeatureMap(int id_1, int id_2, int id_3,
                           float stride, float conf_thres_raw, char scale);
    void NmsProcess(std::vector<cv::Rect2d> &final_bboxes,
                    std::vector<float> &final_scores,
                    std::vector<std::vector<float>> &final_mces);
    int print_info(hbDNNTensorProperties properties);

    //    int find_max_area(Mat &seg_lab, int target_label);
    cv::Point find_mp_inqr(Mat &seg_lab, int target_label);

    float measure_angle();

    int countCategory(const cv::Mat &mat, int category);
    string check_lab(Mat &lab_seg);

    void task_release();
    void perception_release();
};
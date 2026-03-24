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
#include <opencv2/dnn/dnn.hpp>
#include "perception_common.h"

using namespace std;
using namespace cv;


class cdt_perception {
private:
    string model_name;

    int input_count = 0;
    int output_count = 0;

    hbDNNTensor *input;
    hbDNNTensor *output;

    
    std::vector<hbDNNTensor> output_tensors;

    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNTaskHandle_t task_handle = nullptr;
    hbPackedDNNHandle_t packed_dnn_handle = nullptr;

public:
    int order[6];
    int classes_num = 1;
    int preprocess_type=2;
    int reg_param = 16;
    int nms_top_k = 40;
//    float score_threshold=0.15;
    float score_threshold=0.6;
    float nms_threshold = 0.01;

    
    float x_scale=1,  y_scale=1;
    int x_shift=0, y_shift=0;

    int ori_height, ori_width;

    float input_h, input_w;

    hbDNNTensorProperties input_properties;
    std::vector<hbDNNTensor> input_tensors;

    void perception_init(const char *model_file_name);
    void prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    int read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor);
    void perception_preprocess_bgr(Mat &mat);
    void perception_postprocess(std::vector<Detection> &detections);
    void perception_process_bgr(Mat &bgr, std::vector<Detection> &detections);

    void processSmallFeatureMap(hbDNNTensor* cls_tensor, hbDNNTensor* bbox_tensor,
                                std::vector<std::vector<Bbox>>& bboxes,
                                std::vector<std::vector<float>>& scores,
                                int H_8, int W_8);
    void processMediumFeatureMap(hbDNNTensor* cls_tensor, hbDNNTensor* bbox_tensor,
                                 std::vector<std::vector<Bbox>>& bboxes,
                                 std::vector<std::vector<float>>& scores,
                                 int H_16, int W_16);
    void processLargeFeatureMap(hbDNNTensor* cls_tensor, hbDNNTensor* bbox_tensor,
                                std::vector<std::vector<Bbox>>& bboxes,
                                std::vector<std::vector<float>>& scores,
                                int H_32, int W_32);

    std::vector<Detection> post_fix_size(hbDNNTensor* out_tensor);

    void applyNMS(std::vector<std::vector<Bbox>>& bboxes,
                  std::vector<std::vector<float>>& scores,
                  std::vector<std::vector<int>>& indices);
    void task_release();
    void perception_release();
};
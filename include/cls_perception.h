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
// #include "glog/logging.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "perception_common.h"

using namespace std;
using namespace cv;

#define EMPTY ""

//typedef struct Classification {
//    int id;
//    float score;
//    const char *class_name;
//
//    Classification() : class_name(0), id(0), score(0.0) {}
//    Classification(int id, float score, const char *class_name)
//            : id(id), score(score), class_name(class_name) {}
//
//    friend bool operator>(const Classification &lhs, const Classification &rhs) {
//        return (lhs.score > rhs.score);
//    }
//
//    ~Classification() {}
//} Classification;



class cls_perception {
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

    void perception_init(const char *model_file_name);
    void prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor);
    int read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor);
    void get_topk_result(hbDNNTensor *tensor, std::vector<Classification> &top_k_cls, int top_k);

    void perception_process(Mat &mat);
    int perception_postprocess(int top_k);
    int perception_process_bgr(Mat &bgr);
    int perception_postprocess_int64();

    void task_release();
    void perception_release();
};
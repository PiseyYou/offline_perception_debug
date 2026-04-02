/**************************************************************
 *  Copyright (c):   LF Intelligent Technology Co., LTD.
 *  Department:  Perception
 *  Description: dsg_perception class
 *
 *  @author:     YouFeng
 *  @data        2026/01/15 下午4:01
 **************************************************************/
#include <iostream>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "perception_common.h"
#include "dsg_perception.h"


using namespace std;
using namespace cv;

int dsg_perception::BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
    auto height = bgr_mat.rows;
    auto width = bgr_mat.cols;

    if (height % 2 || width % 2) {
        std::cerr << "input img height and width must aligned by 2!";
        return -1;
    }
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    if (yuv_mat.data == nullptr) {
        std::cerr << "yuv_mat.data is null pointer" << std::endl;
        return -1;
    }

    auto *yuv = yuv_mat.ptr<uint8_t>();
    if (yuv == nullptr) {
        std::cerr << "yuv is null pointer" << std::endl;
        return -1;
    }
    img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
    auto *ynv12 = img_nv12.ptr<uint8_t>();

    int32_t uv_height = height / 2;
    int32_t uv_width = width / 2;

    // copy y data
    int32_t y_size = height * width;
    memcpy(ynv12, yuv, y_size);

    // copy uv data
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return 0;
}

void dsg_perception::perception_init(const char *model_file_name) {

        // this->num_classes=type;
        int ret = hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
        if (ret != 0) {
            std::cerr << "[DSG] ERROR: hbDNNInitializeFromFiles failed, ret=" << ret
                      << ", model=" << model_file_name << std::endl;
            return;
        }

        const char **model_name_list = nullptr;
        int model_count = 0;
        ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
        if (ret != 0 || model_name_list == nullptr || model_count == 0) {
            std::cerr << "[DSG] ERROR: hbDNNGetModelNameList failed, ret=" << ret
                      << ", model_count=" << model_count << std::endl;
            return;
        }

        ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
        if (ret != 0) {
            std::cerr << "[DSG] ERROR: hbDNNGetModelHandle failed, ret=" << ret << std::endl;
            return;
        }
//        printf("Model info:\nmodel_name: %s\n", model_name_list[0]);

//        this->packed_dnn_handle=packed_dnn_handle;
//        print_model_info(packed_dnn_handle);

        int input_count = 0;
        int output_count = 0;

        std::vector<hbDNNTensor> input_tensors;
        std::vector<hbDNNTensor> output_tensors;

        {
            int inCount_ret = hbDNNGetInputCount(&input_count, dnn_handle);
            int outCount_ret = hbDNNGetOutputCount(&output_count, dnn_handle);
            if(inCount_ret != 0 || outCount_ret != 0) {
                std::cerr << "[DSG] ERROR: hbDNNGetInputCount/hbDNNGetOutputCount failed, "
                          << "inCount_ret=" << inCount_ret << ", outCount_ret=" << outCount_ret << std::endl;
                return;
            }

            input_tensors.resize(input_count);
            output_tensors.resize(output_count);
            ret = prepare_tensor(input_tensors.data(), output_tensors.data());
            if (ret != 0) {
                std::cerr << "[DSG] ERROR: prepare_tensor failed, ret=" << ret << std::endl;
                return;
            }
        }

        this->input_count=input_count;
        this->output_count=output_count;

        this->input_tensors = input_tensors;
        this->output_tensors = output_tensors;

        std::cout << "[DSG] Model initialized successfully: " << model_file_name << std::endl;
}

int dsg_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor) {
    int ret = hbDNNGetInputCount(&input_count, dnn_handle);
    if (ret != 0) {
        std::cerr << "[DSG] ERROR: hbDNNGetInputCount failed in prepare_tensor, ret=" << ret << std::endl;
        return -1;
    }
    ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    if (ret != 0) {
        std::cerr << "[DSG] ERROR: hbDNNGetOutputCount failed in prepare_tensor, ret=" << ret << std::endl;
        return -1;
    }

    this->input = input_tensor;
    for (int i = 0; i < input_count; i++) {

        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
        if (inTensor_ret != 0) {
            std::cerr << "[DSG] ERROR: hbDNNGetInputTensorProperties failed, i=" << i << ", ret=" << inTensor_ret << std::endl;
            return -1;
        }
//        cout << "model h/w: " << input[i].properties.validShape.dimensionSize[2] << "/" << input[i].properties.validShape.dimensionSize[3] << endl;
        int input_memSize = input[i].properties.alignedByteSize;
        int input_allocCached_ret = hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
        if (input_allocCached_ret != 0) {
            std::cerr << "[DSG] ERROR: hbSysAllocCachedMem failed for input, i=" << i
                      << ", size=" << input_memSize << ", ret=" << input_allocCached_ret << std::endl;
            return -1;
        }
        input[i].properties.alignedShape = input[i].properties.validShape;
    }

    this->output = output_tensor;
    for (int i = 0; i < output_count; i++) {
        int outTensor_ret = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        if (outTensor_ret != 0) {
            std::cerr << "[DSG] ERROR: hbDNNGetOutputTensorProperties failed, i=" << i << ", ret=" << outTensor_ret << std::endl;
            return -1;
        }
        int output_memSize = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);
        if (output_allocCached_ret != 0) {
            std::cerr << "[DSG] ERROR: hbSysAllocCachedMem failed for output, i=" << i
                      << ", size=" << output_memSize << ", ret=" << output_allocCached_ret << std::endl;
            return -1;
        }

    }
    return 0;
}

int dsg_perception::prepare_mat_nv12(Mat originMat) {
//    auto start4 = std::chrono::high_resolution_clock::now();
    hbDNNTensor *input = input_tensors.data();
    hbDNNTensorProperties Properties = input->properties;
    int input_h = Properties.validShape.dimensionSize[1];
    int input_w = Properties.validShape.dimensionSize[2];
    if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        input_h = Properties.validShape.dimensionSize[2];
        input_w = Properties.validShape.dimensionSize[3];
    }

    // resize
    cv::Mat mat;
    mat.create(input_h, input_w, originMat.type());
    cv::resize(originMat, mat, mat.size(), 0, 0);
//    cout << "w/h: " << mat.cols << " " << mat.rows << endl;
    // convert to YUV420
    if (input_h % 2 || input_w % 2) {
        cout << "input img height and width must aligned by 2!" << endl;
        return -1;
    }
    cv::Mat yuv_mat;
    cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);

//    string dstImg = "res_" + to_string(tcount) +".jpg";
//    cv::imwrite(dstImg, yuv_mat);
    uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

    // copy y data
    auto data = input->sysMem[0].virAddr;
    int32_t y_size = input_h * input_w;
    memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

    // copy uv data
    int32_t uv_height = input_h / 2;
    int32_t uv_width = input_w / 2;
    uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
    uint8_t *u_data = nv12_data + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return 0;
};

void dsg_perception::perception_process(Mat &mat) {
    int nv12Ret = prepare_mat_nv12(mat);
    for (int j = 0; j < input_count; j++) {
        hbSysFlushMem(&input_tensors[j].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    this->output=output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    infer_ctrl_param.bpuCoreId = 0;
    int infer_ret = hbDNNInfer(&task_handle,
                               &output,
                               input_tensors.data(),
                               dnn_handle,
                               &infer_ctrl_param);
    if(infer_ret!=0) cout<< "hbDNNInfer failed" << endl;
    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    if(task_ret!=0) cout << "hbDNNWaitTaskDone failed" << endl;
}



void dsg_perception::perception_postprocess_match(Mat& lab_out) {
    hbSysFlushMem(&(output_tensors[0].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // Simplified version: directly get result and draw
    // Mat result = argmax_and_draw(mat, count);
    lab_match(output_tensors.data(), lab_out);
}

void dsg_perception::process_infer_match(Mat &mat, Mat& lab_out)
{
    perception_process(mat);
    perception_postprocess_match(lab_out);
    task_release();
}


// Simplified argmax that returns cv::Mat directly (single channel, values 20,21,22)
void dsg_perception::lab_match(hbDNNTensor *output_tensors, Mat& lab_out) {
    hbDNNTensor *tensors = output_tensors;
    int h_index, w_index, c_index;
    get_tensor_hwc_index(&tensors[0], &h_index, &w_index, &c_index);
    int height = tensors[0].properties.validShape.dimensionSize[h_index];
    int width = tensors[0].properties.validShape.dimensionSize[w_index];
    // int channel = tensors[0].properties.validShape.dimensionSize[c_index];

    // cout << "out_tensor: h*w*c: " << height << "/" << width << "/" << channel << endl;
    // cout << "tensorLayout: " << tensors[0].properties.tensorLayout << " (NCHW=2, NHWC=0)" << endl;

    // Create result Mat (single channel, CV_8UC1)
    // cv::Mat lab_out(height, width, CV_8UC1);
    uint8_t *result_ptr = lab_out.ptr<uint8_t>();

    int8_t *data = reinterpret_cast<int8_t *>(tensors[0].sysMem[0].virAddr);

    // Python: pred = np.argmax(output_buffer[0], axis=0)
    if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        // NCHW format: data layout is [C][H][W]
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float top_score = -1000000.0f;
                // int top_index = data[h * width + w];
                int8_t *top_index = data+ (h * width + w)*4;
                uint8_t match_label;
                // 映射表：根据 top_index[0] 的值设置 match_label
                if (top_index[0] == 0) {
                    match_label = 1;
                } else if (top_index[0] == 1) {
                    match_label = 5;
                } else if (top_index[0] == 2) {
                    match_label = 3;
                } else {
                    match_label = 1;
                }
                result_ptr[h * width + w] = match_label;
            }
        }
    }
}

int dsg_perception::get_tensor_hwc_index(hbDNNTensor *tensor, int *h_index, int *w_index, int *c_index) {
    if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        *h_index = 1;
        *w_index = 2;
        *c_index = 3;
    } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        *c_index = 1;
        *h_index = 2;
        *w_index = 3;
    } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NONE) {
        *c_index = 1;
        *h_index = 2;
        *w_index = 3;
    } else {
        return -1;
    }
    return 0;
}

void dsg_perception::task_release(){
    int res_ret = hbDNNReleaseTask(task_handle);
    if(res_ret!=0) cout<< "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void dsg_perception::perception_release() {
    for (int i = 0; i < input_count; i++) {
        int infree_ret = hbSysFreeMem(&(input_tensors[i].sysMem[0]));
        if(infree_ret!=0) cout << "hbSysFreeMem failed"  <<endl;
    }
    for (int i = 0; i < output_count; i++) {
        int outfree_ret = hbSysFreeMem(&(output_tensors[i].sysMem[0]));
        if(outfree_ret!=0) cout<< "hbSysFreeMem failed" << endl;
    }
    int dnn_ret = hbDNNRelease(packed_dnn_handle);
    if(dnn_ret!=0) cout << "hbDNNRelease failed" << endl;

}
//
// Created by youfeng on 2026/1/15.
//

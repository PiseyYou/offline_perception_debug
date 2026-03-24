#include "cdt_perception.h"
#include <chrono>

//std::vector<std::string> class_names = {"cable"};

void cdt_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor) {
    /** Tips:
     * For input memory size:
     * *   input_memSize = input[i].properties.alignedByteSize
     * For output memory size:
     * *   output_memSize = output[i].properties.alignedByteSize
     */
    this->input = input_tensor;
    input->properties = input_properties;

    int input_memSize = input->properties.alignedByteSize;
    int input_allocCached_ret = hbSysAllocCachedMem(&input->sysMem[0], input_memSize);

    if (input_allocCached_ret != 0)
        cout << "hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed" << endl;

    this->output = output_tensor;
    for (int i = 0; i < output_count; i++) {
        int outTensor_ret = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int output_memSize = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);
        if (outTensor_ret || output_allocCached_ret != 0)
            cout << "hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed" << endl;
    }

    int32_t H_8 = static_cast<int32_t>(input_h / 8);
    int32_t H_16 = static_cast<int32_t>(input_h / 16);
    int32_t H_32 = static_cast<int32_t>(input_h / 32);
    int32_t W_8 = static_cast<int32_t>(input_w / 8);
    int32_t W_16 = static_cast<int32_t>(input_w / 16);
    int32_t W_32 = static_cast<int32_t>(input_w / 32);
    int32_t order_we_want[6][3] = {
            {H_8, W_8, classes_num},   // output[order[3]]: (1, H // 8,  W // 8,  classes_num)
            {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
            {H_16, W_16, classes_num}, // output[order[4]]: (1, H // 16, W // 16, classes_num)
            {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16,  64)
            {H_32, W_32, classes_num}, // output[order[5]]: (1, H // 32, W // 32, classes_num)
            {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32,  64)
    };

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            hbDNNTensorProperties output_properties;
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j);
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2]) {
                order[i] = j;
                break;
            }
        }
    }
}


int cdt_perception::read_image_2_tensor_as_nv12(Mat &bgr_mat,
                                    hbDNNTensor *input_tensor) {
    input = input_tensor;

    cv::Mat yuv_mat;
    cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

    // 拷贝 Y 数据
    auto data = input->sysMem[0].virAddr;
    int32_t y_size = input_h * input_w;
    memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

    // 拷贝 UV 数据
    int32_t uv_height = input_h / 2;
    int32_t uv_width = input_w / 2;
    uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
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

void cdt_perception::perception_init(const char *model_file_name){

    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    cout << "DNN runtime version: " << hbDNNGetVersion() << endl;

    int inCount_ret = hbDNNGetInputCount(&input_count, dnn_handle);
    int outCount_ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    if (inCount_ret || outCount_ret != 0) cout << "hbDhNNGetInputCount/hbDNNGetOutputCount failed" << endl;

    input_tensors.resize(input_count);
    output_tensors.resize(output_count);

    hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0);
    input_h = input_properties.validShape.dimensionSize[2];
    input_w = input_properties.validShape.dimensionSize[3];
    prepare_tensor(input_tensors.data(), output_tensors.data());
}

void cdt_perception::perception_preprocess_bgr(Mat &mat) {

    int nv12Ret = read_image_2_tensor_as_nv12(mat, input_tensors.data());
    if (nv12Ret != 0) cout << "read_image_2_tensor_as_nv12 failed" << endl;

    for (int j = 0; j < this->input_count; j++) {
        hbSysFlushMem(&input_tensors[j].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    output = output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    infer_ctrl_param.bpuCoreId = 0;
    int infer_ret = hbDNNInfer(&task_handle,
                               &output,
                               input_tensors.data(),
                               dnn_handle,
                               &infer_ctrl_param);
    if (infer_ret != 0) cout << "hbDNNInfer failed" << endl;
    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    if (task_ret != 0) cout << "hbDNNWaitTaskDone failed" << endl;
};


void cdt_perception::applyNMS(std::vector<std::vector<Bbox>>& bboxes,
                              std::vector<std::vector<float>>& scores,
                              std::vector<std::vector<int>>& indices) {
    for (int i = 0; i < classes_num; i++) {
        std::vector<cv::Rect> cv_bboxes;
        for (const auto& bbox : bboxes[i]) {
            cv_bboxes.push_back(cv::Rect(bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin));
        }
        cv::dnn::NMSBoxes(cv_bboxes, scores[i], score_threshold, nms_threshold, indices[i], 1.f, nms_top_k);
    }
}


void cdt_perception::processSmallFeatureMap(hbDNNTensor* cls_tensor, hbDNNTensor* bbox_tensor,
                                            std::vector<std::vector<Bbox>>& bboxes,
                                            std::vector<std::vector<float>>& scores,
                                            int H_8, int W_8) {
    if (cls_tensor->properties.quantiType != NONE) {
        std::cout << "output[order[0]] QuantiType is not NONE, please check!" << std::endl;
        return;
    }
    if (bbox_tensor->properties.quantiType != SCALE) {
        std::cout << "output[order[1]] QuantiType is not SCALE, please check!" << std::endl;
        return;
    }

    hbSysFlushMem(&(cls_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    auto* s_cls_raw = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
    auto* s_bbox_raw = reinterpret_cast<int32_t*>(bbox_tensor->sysMem[0].virAddr);
    auto* s_bbox_scale = reinterpret_cast<float*>(bbox_tensor->properties.scale.scaleData);

    float CONF_THRES_RAW = -log(1 / score_threshold - 1);

    for (int h = 0; h < H_8; h++) {
        for (int w = 0; w < W_8; w++) {
            float* cur_s_cls_raw = s_cls_raw;
            int32_t* cur_s_bbox_raw = s_bbox_raw;

            int cls_id = 0;
            for (int i = 1; i < classes_num; i++) {
                if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }

            if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW) {
                s_cls_raw += classes_num;
                s_bbox_raw += reg_param * 4;
                continue;
            }

            float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++) {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < reg_param; j++) {
                    int index_id = reg_param * i + j;
                    dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                s_cls_raw += classes_num;
                s_bbox_raw += reg_param * 4;
                continue;
            }

            float x1 = (w + 0.5f - ltrb[0]) * 8.0f;
            float y1 = (h + 0.5f - ltrb[1]) * 8.0f;
            float x2 = (w + 0.5f + ltrb[2]) * 8.0f;
            float y2 = (h + 0.5f + ltrb[3]) * 8.0f;

            Bbox bbox(x1, y1, x2, y2);
            bboxes[cls_id].push_back(bbox);
            scores[cls_id].push_back(score);

            s_cls_raw += classes_num;
            s_bbox_raw += reg_param * 4;
        }
    }
}


void cdt_perception::processMediumFeatureMap(hbDNNTensor* cls_tensor, hbDNNTensor* bbox_tensor,
                                             std::vector<std::vector<Bbox>>& bboxes,
                                             std::vector<std::vector<float>>& scores,
                                             int H_16, int W_16) {
    if (cls_tensor->properties.quantiType != NONE) {
        std::cout << "output[order[2]] QuantiType is not NONE, please check!" << std::endl;
        return;
    }
    if (bbox_tensor->properties.quantiType != SCALE) {
        std::cout << "output[order[3]] QuantiType is not SCALE, please check!" << std::endl;
        return;
    }

    hbSysFlushMem(&(cls_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    auto* m_cls_raw = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
    auto* m_bbox_raw = reinterpret_cast<int32_t*>(bbox_tensor->sysMem[0].virAddr);
    auto* m_bbox_scale = reinterpret_cast<float*>(bbox_tensor->properties.scale.scaleData);

    float CONF_THRES_RAW = -log(1 / score_threshold - 1);

    for (int h = 0; h < H_16; h++) {
        for (int w = 0; w < W_16; w++) {
            float* cur_m_cls_raw = m_cls_raw;
            int32_t* cur_m_bbox_raw = m_bbox_raw;

            int cls_id = 0;
            for (int i = 1; i < classes_num; i++) {
                if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }

            if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW) {
                m_cls_raw += classes_num;
                m_bbox_raw += reg_param * 4;
                continue;
            }

            float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++) {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < reg_param; j++) {
                    int index_id = reg_param * i + j;
                    dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                m_cls_raw += classes_num;
                m_bbox_raw += reg_param * 4;
                continue;
            }

            float x1 = (w + 0.5f - ltrb[0]) * 16.0f;
            float y1 = (h + 0.5f - ltrb[1]) * 16.0f;
            float x2 = (w + 0.5f + ltrb[2]) * 16.0f;
            float y2 = (h + 0.5f + ltrb[3]) * 16.0f;

            Bbox bbox(x1, y1, x2, y2);
            bboxes[cls_id].push_back(bbox);
            scores[cls_id].push_back(score);

            m_cls_raw += classes_num;
            m_bbox_raw += reg_param * 4;
        }
    }
}


void cdt_perception::processLargeFeatureMap(hbDNNTensor* cls_tensor, hbDNNTensor* bbox_tensor,
                                            std::vector<std::vector<Bbox>>& bboxes,
                                            std::vector<std::vector<float>>& scores,
                                            int H_32, int W_32) {
    if (cls_tensor->properties.quantiType != NONE) {
        std::cout << "output[order[4]] QuantiType is not NONE, please check!" << std::endl;
        return;
    }
    if (bbox_tensor->properties.quantiType != SCALE) {
        std::cout << "output[order[5]] QuantiType is not SCALE, please check!" << std::endl;
        return;
    }

    hbSysFlushMem(&(cls_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    auto* l_cls_raw = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
    auto* l_bbox_raw = reinterpret_cast<int32_t*>(bbox_tensor->sysMem[0].virAddr);
    auto* l_bbox_scale = reinterpret_cast<float*>(bbox_tensor->properties.scale.scaleData);

    float CONF_THRES_RAW = -log(1 / score_threshold - 1);

    for (int h = 0; h < H_32; h++) {
        for (int w = 0; w < W_32; w++) {
            float* cur_l_cls_raw = l_cls_raw;
            int32_t* cur_l_bbox_raw = l_bbox_raw;

            int cls_id = 0;
            for (int i = 1; i < classes_num; i++) {
                if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }

            if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW) {
                l_cls_raw += classes_num;
                l_bbox_raw += reg_param * 4;
                continue;
            }

            float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++) {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < reg_param; j++) {
                    int index_id = reg_param * i + j;
                    dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                l_cls_raw += classes_num;
                l_bbox_raw += reg_param * 4;
                continue;
            }

            float x1 = (w + 0.5f - ltrb[0]) * 32.0f;
            float y1 = (h + 0.5f - ltrb[1]) * 32.0f;
            float x2 = (w + 0.5f + ltrb[2]) * 32.0f;
            float y2 = (h + 0.5f + ltrb[3]) * 32.0f;

            Bbox bbox(x1, y1, x2, y2);
            bboxes[cls_id].push_back(bbox);
            scores[cls_id].push_back(score);

            l_cls_raw += classes_num;
            l_bbox_raw += reg_param * 4;
        }
    }
}

std::vector<Detection> cdt_perception::post_fix_size(hbDNNTensor* out_tensor){
    // 初始化存储检测结果的容器
    std::vector<std::vector<Bbox>> bboxes(classes_num);
    std::vector<std::vector<float>> scores(classes_num);

    // 处理三个尺度的特征图
    int32_t H_8 = static_cast<int32_t>(input_h / 8);
    int32_t W_8 = static_cast<int32_t>(input_w / 8);
    int32_t H_16 = static_cast<int32_t>(input_h / 16);
    int32_t W_16 = static_cast<int32_t>(input_w / 16);
    int32_t H_32 = static_cast<int32_t>(input_h / 32);
    int32_t W_32 = static_cast<int32_t>(input_w / 32);

    // 处理小目标特征图
    processSmallFeatureMap(&out_tensor[order[0]], &out_tensor[order[1]], bboxes, scores, H_8, W_8);

    // 处理中目标特征图
    processMediumFeatureMap(&out_tensor[order[2]], &out_tensor[order[3]], bboxes, scores, H_16, W_16);

    // 处理大目标特征图
    processLargeFeatureMap(&out_tensor[order[4]], &out_tensor[order[5]], bboxes, scores, H_32, W_32);

    // 应用NMS
    std::vector<std::vector<int>> indices(classes_num);
    applyNMS(bboxes, scores, indices);

    // 构建最终检测结果
    std::vector<Detection> results;
    for (int cls_id = 0; cls_id < classes_num; cls_id++) {
        for (std::vector<int>::iterator it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it) {
            float x1 = (bboxes[cls_id][*it].xmin - x_shift) / x_scale;
            float y1 = (bboxes[cls_id][*it].ymin - y_shift) / y_scale;
            float x2 = (bboxes[cls_id][*it].xmax - x_shift) / x_scale;
            float y2 = (bboxes[cls_id][*it].ymax - y_shift) / y_scale;
            if(y2<380) continue;
            cout <<"x1/y1/x2/y2: " << x1 << "/" << y1 << "/" << x2 << "/" << y2 << endl;
            Bbox bbox(x1, y1, x2, y2);
//            Detection detection(cls_id, scores[cls_id][*it], bbox, class_names[cls_id % class_names.size()].c_str());
            Detection detection(cls_id, scores[cls_id][*it], bbox, "cable_tie");
            results.push_back(detection);
        }
    }
    cout <<"results.size(): " << results.size() << endl;
    return results;
}

void cdt_perception::perception_postprocess(std::vector<Detection> &detections) {
    auto perception = std::shared_ptr<Perception>(new Perception);
    perception->type = Perception::DET;

    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // cv::Mat result_mat(ori_height, ori_width, CV_8UC1);
//    detections = postProcess(output, y_scale, x_scale, y_shift, x_shift);
    detections = post_fix_size(output);
}


void cdt_perception::perception_process_bgr(Mat &bgr, std::vector<Detection> &detections){
    // cout <<"===before perception_preprocess_bgr===" << endl;
    perception_preprocess_bgr(bgr);
    // cout <<"===before perception_postprocess===" << endl;
    perception_postprocess(detections);
    // cout <<"===before task_release===" << endl;
    task_release();
}

void cdt_perception::task_release() {
    int res_ret = hbDNNReleaseTask(task_handle);
    if (res_ret != 0) cout << "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void cdt_perception::perception_release() {
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
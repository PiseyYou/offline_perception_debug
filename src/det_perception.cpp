/**************************************************************
 *  Copyright (c):   LF Intelligent Technology Co., LTD.
 *  Department:  Perception
 *  Description: bup process to get bounding box
 *
 *  @author:     YouFeng
 *  @data        2024/09/29 下午16:44
 **************************************************************/

#include "det_perception.h"

//DetConfig nanodet_ptq_config = {
//        {{8, 16, 32, 64}},
//        9,
//        {"person", "animal", "obstacle", "shoes", "wheel", "leaf debris", "faeces",
//         "rock", "background"}
//};

int det_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor) {
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);

    this->input = input_tensor;
    for (int i = 0; i < input_count; i++) {
        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
        this->model_height = (input[i].properties).validShape.dimensionSize[2];
        this->model_width = (input[i].properties).validShape.dimensionSize[3];
//        cout << "model_height/model_width: " << model_height << "/" << model_width << endl;
//        LOG_INFO("model_height/model_width: {}/{}", model_height, model_width);
        int input_memSize = input[i].properties.alignedByteSize;
        int input_allocCached_ret = hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
        input[i].properties.alignedShape = input[i].properties.validShape;

        if(inTensor_ret || input_allocCached_ret !=0)
//            LOG_ERROR("hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed");
            cout << "hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed" << endl;
    }

//    hbDNNTensor *output = output_tensor;
    this->output = output_tensor;
    for (int i = 0; i < output_count; i++) {
        int outTensor_ret = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int output_memSize = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);
        if(outTensor_ret || output_allocCached_ret !=0)
//            LOG_ERROR("hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed");
            cout<< "hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed" << endl;
    }
    return 0;
}

int det_perception::read_image_2_tensor_as_nv12(Mat &bgr_mat, hbDNNTensor *input_tensor) {
    hbDNNTensor *input = input_tensor;
    hbDNNTensorProperties Properties = input->properties;
//    int tensor_id = 0;
    int input_h = Properties.validShape.dimensionSize[1];
    int input_w = Properties.validShape.dimensionSize[2];
    if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        input_h = Properties.validShape.dimensionSize[2];
        input_w = Properties.validShape.dimensionSize[3];
    }

    if (bgr_mat.empty()) {
        cout << "image file not exist!" << endl;
        return -1;
    }
    // resize
    cv::Mat mat;
    mat.create(input_h, input_w, bgr_mat.type());
    cv::resize(bgr_mat, mat, mat.size(), 0, 0);
    // convert to YUV420
    if (input_h % 2 || input_w % 2) {
        cout << "input img height and width must aligned by 2!" << endl;
        return -1;
    }
    cv::Mat yuv_mat;
    cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
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
        if (u_data && v_data) {
            *nv12++ = *u_data++;
            *nv12++ = *v_data++;
        }
    }
    return 0;
}

int det_perception::read_nv12(Mat &yuv_mat, hbDNNTensor *input_tensor) {
    hbDNNTensor *input = input_tensor;
//    hbDNNTensorProperties Properties = input->properties;
//    int tensor_id = 0;
    if (yuv_mat.empty()) {
        cout << "image file not exist!" << endl;
        return -1;
    }

    int yuv_height = this->model_height*3/2;
    int yuv_width = this->model_width;
    int yuv_size = yuv_height*yuv_width;

    uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();
    auto data = input->sysMem[0].virAddr;
    memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, yuv_size);
    return 0;
}


void det_perception::perception_init(const char *model_file_name, float dect_threshold) {

//    this->det_config_ = nanodet_ptq_config;
    // cout << "nanodet_ptq_config is load..." << endl;
//    LOG_INFO("nanodet_ptq_config is load...");
//    cout << "nanodet_ptq_config is load..." << endl;

//    map<int, int> = {100: 4, 101: 4, 102:1, 103:1, 104:1, 105:1, 106:1, 107:1, 108:1};
//    det_label_match[100] = 4;
//    det_label_match[101] = 4;
//    det_label_match[102] = 1;
//    det_label_match[103] = 1;
//    det_label_match[104] = 1;
//    det_label_match[105] = 1;
//    det_label_match[106] = 1;
//    det_label_match[107] = 1;
//    det_label_match[108] = 1;

    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    {
        int inCount_ret = hbDNNGetInputCount(&this->input_count, dnn_handle);
        int outCount_ret = hbDNNGetOutputCount(&this->output_count, dnn_handle);
        if (inCount_ret || outCount_ret != 0) cout << "hbDhNNGetInputCount/hbDNNGetOutputCount failed" << endl;
//        cout << "inCount_ret/outCount_ret " << inCount_ret << "/" << outCount_ret << endl;

        this->input_tensors.resize(this->input_count);
        this->output_tensors.resize(this->output_count);
        prepare_tensor(this->input_tensors.data(), this->output_tensors.data());
    }

    m_conf_threshold =dect_threshold;
}

float det_perception::det_sigmoid(const float input) {
    return 1.f / (1.f + expf(-input));
}

void det_perception::perception_process(Mat &mat) {
    int nv12Ret = read_image_2_tensor_as_nv12(mat, input_tensors.data());
    if (nv12Ret != 0){
//        LOG_ERROR("read_image_2_tensor_as_nv12 faile");
        cout << "read_image_2_tensor_as_nv12 faile" << endl;
    }
//    std::cout << "==read_image_2_tensor_as_nv12"  << std::endl;
    for (int j = 0; j < this->input_count; j++) {
        hbSysFlushMem(&input_tensors[j].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    this->output = output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    infer_ctrl_param.bpuCoreId = 0;
    int infer_ret = hbDNNInfer(&task_handle,
                               &output,
                               input_tensors.data(),
                               dnn_handle,
                               &infer_ctrl_param);
    if (infer_ret != 0){
        cout << "hbDNNInfer faile" << endl;
    }
    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    if (task_ret != 0){
        cout << "hbDNNWaitTaskDone faile" << endl;
    }
}

void det_perception::perception_process_nv12(Mat &yuv_mat) {
    int nv12Ret = read_nv12(yuv_mat, input_tensors.data());
    if (nv12Ret != 0){
        cout << "det_perception: read_nv12 faile, return " << nv12Ret << endl;
    }
    for (int j = 0; j < this->input_count; j++) {
        hbSysFlushMem(&input_tensors[j].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    this->output = output_tensors.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    infer_ctrl_param.bpuCoreId = 0;
    int infer_ret = hbDNNInfer(&task_handle,
                               &output,
                               input_tensors.data(),
                               dnn_handle,
                               &infer_ctrl_param);
    if (infer_ret != 0){
//        LOG_ERROR("det_perception: do hbDNNInfer faile, return {}",infer_ret);
        cout << "det_perception: do hbDNNInfer faile, return " << infer_ret << endl;
    }

//    LOG_INFO("det_perception: do hbDNNInfer finish");
    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    cout << "det_perception: finish hbDNNWaitTaskDone" << endl;
    if (task_ret != 0){
        cout << "det_perception:hbDNNWaitTaskDone failed, return " << task_ret << endl;
    }
}



// mat here refers to the original image before padding/resizing
void det_perception::perception_postprocess_nanodet(Mat mat, vector<Detection> &picDet) {
    auto perception = std::shared_ptr<Perception>(new Perception);
    perception->type = Perception::DET;

    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    vector<Detection> dets = decode_nanodet(output_tensors.data());
    std::vector<Detection> result;
    nms(dets, iou_threshold, topk, result, false);

//    refine_rect(result, mat, picDet);
    refine_rect(result, picDet);
//    cout << "picDet size is " <<  picDet.size() << endl;
}

void det_perception::perception_postprocess_nanodet_nv12(Mat yuv_mat, vector<Detection> &picDet) {
    auto perception = std::shared_ptr<Perception>(new Perception);
    perception->type = Perception::DET;

    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    vector<Detection> dets = decode_nanodet(output_tensors.data());
    std::vector<Detection> result;
    nms(dets, iou_threshold, topk, result, false);

//    refine_rect_nv12(result, yuv_mat, picDet);
    refine_rect_nv12(result, picDet);
    // LOG_INFO("nv12 picDet size is {}", picDet.size());
}

vector<Detection> det_perception::refine_rect_nv12(std::vector<Detection> det, vector<Detection> &picDet) {

    for (size_t i = 0; i < det.size(); i++) {
        Bbox &bbox = det[i].bbox;
        float wscale = YJ_MODEL_OUTPUT_WIDTH/float(this->model_width);
        float hscale = YJ_MODEL_OUTPUE_HEIGHT/float(this->model_height);
        // cout << "wscale/hscale: " << wscale << "/" << hscale << endl;

        bbox.xmin = bbox.xmin * wscale;
        bbox.xmax = bbox.xmax * wscale;
        bbox.ymin = bbox.ymin * hscale;
        bbox.ymax = bbox.ymax * hscale;

        if(bbox.xmin<0 || bbox.xmin>YJ_MODEL_OUTPUT_WIDTH) continue;
        if(bbox.xmax<0 || bbox.xmax>YJ_MODEL_OUTPUT_WIDTH) continue;
        if(bbox.ymin<0 || bbox.ymin>YJ_MODEL_OUTPUE_HEIGHT) continue;
        if(bbox.ymax<0 || bbox.ymax>YJ_MODEL_OUTPUE_HEIGHT) continue;

        picDet.emplace_back(det[i]);
    }
    return picDet;
}

vector<Detection> det_perception::decode_nanodet(hbDNNTensor *output_tensors) {
    std::vector<Detection> dets;
    int strides[4] = {8, 16, 32, 64};
    // float conf_threshold=-0.5f;

//    auto start = std::chrono::high_resolution_clock::now();
    for (int b=0; b<4; b++){
        hbDNNTensor *intensor = &output_tensors[b];
        int *shape = intensor->properties.validShape.dimensionSize;
        auto data = reinterpret_cast<float *>(intensor->sysMem[0].virAddr);
        decode_nanodet_nhwc(
                // Input network output
                data,
                // Inputs parameter
                strides[b],
                shape[1],
                shape[2],
                m_conf_threshold,
                // Outputs
                dets);
    }
    return dets;
}

float det_perception::softmax8(const float * input){
    float no = 0.;
    float dno = 0.;
    for (int i=0; i<8;i++){
        float ex = expf(input[i]);
        no+=ex*((float)i);
        dno+=ex;
    }
    return no/dno;
}


void det_perception::decode_nanodet_nhwc(
        // Input network output
        const float * outputs,
        // Inputs parameter
        const int stride,
        const int h,
        const int w,
        const float conf_threshold,
        // Outputs
        std::vector<Detection> &dets
){
    int size = h*w;
    const float stridef = (float) stride;
    for(int i=0; i<size; i++){
        // Calculate the x, y location
        const int y = i / w;
        const int x = i % w;

        const float * cls = &outputs[(y* w + x) *NUM_PARAM];
        const float * reg = &outputs[(y* w + x) *NUM_PARAM + NUM_CLASS_NANO];

        // fetch the max score (class types)
        // neonable?
        float conf = -1000000.f;
        int cindex = 0;
        for (int c=0; c<NUM_CLASS_NANO; c++){
            const float score = cls[c];  // hwc
            if (score>conf){
                conf = score;
                cindex = c;
            }
        }

        if (det_sigmoid(conf)>conf_threshold){
            Detection detection;
            // convert the box
            const float l = softmax8(&reg[0]) * stridef;
            const float t = softmax8(&reg[8]) * stridef;
            const float r = softmax8(&reg[16]) * stridef;
            const float b = softmax8(&reg[24]) * stridef;

            const float yf = ((float)(y)) * stridef;
            const float xf = ((float)(x)) * stridef;

            // Write to the output data while also doing a transpose
            detection.bbox.xmin = xf - l;
            detection.bbox.ymin = yf - t;
            detection.bbox.xmax = xf + r;
            detection.bbox.ymax = yf + b;
            detection.area = (b+t)*(l+r); // this is for nms calculation
            detection.skip = 0;
            detection.id = cindex;
            detection.score = det_sigmoid(conf);
            // int temp = det_sigmoid(conf)*100;
            // detection.score = float(temp)/100;

            dets.push_back(detection);
        }
    }
}

vector<Detection> det_perception::refine_rect(std::vector<Detection> det, vector<Detection> &picDet) {
    for (size_t i = 0; i < det.size(); i++) {
        // auto &color = colors[det[i].id % 7];
        Bbox &bbox = det[i].bbox;

//        float wscale = mat.cols/float(this->model_width);
//        float hscale = mat.rows/float(this->model_height);

        float wscale = YJ_MODEL_OUTPUT_WIDTH/float(this->model_width);
        float hscale = YJ_MODEL_OUTPUE_HEIGHT/float(this->model_height);

        bbox.xmin = bbox.xmin * wscale;
        bbox.xmax = bbox.xmax * wscale;
        bbox.ymin = bbox.ymin * hscale;
        bbox.ymax = bbox.ymax * hscale;

        if(bbox.xmin<0 || bbox.xmin>YJ_MODEL_OUTPUT_WIDTH) continue;
        if(bbox.xmax<0 || bbox.xmax>YJ_MODEL_OUTPUT_WIDTH) continue;
        if(bbox.ymin<0 || bbox.ymin>YJ_MODEL_OUTPUE_HEIGHT) continue;
        if(bbox.ymax<0 || bbox.ymax>YJ_MODEL_OUTPUE_HEIGHT) continue;

        picDet.emplace_back(det[i]);
    }
    return picDet;
}


void det_perception::nms(std::vector<Detection> &input,
                   float iou_threshold,
                   int top_k,
                   std::vector<Detection> &result,
                   bool suppress) {
    // sort order by score desc
    // sort order by score desc
    std::stable_sort(input.begin(), input.end(), std::greater<Detection>());
    if (input.size() > NMS_MAX_INPUT) {
        input.resize(NMS_MAX_INPUT);
    }

    int count = 0;
    for (size_t i = 0; count < top_k && i < input.size(); i++) {

        if (input[i].skip > 0) {
            continue;
        }
        input[i].skip = 1;
        ++count;

        for (size_t j = i + 1; j < input.size(); ++j) {

            if (input[j].skip>0){
                continue;
            }
            if (suppress == false) {
                if (input[i].id != input[j].id) {
                    continue;
                }
            }

            // intersection area
            float xx1 = std::max(input[i].bbox.xmin, input[j].bbox.xmin);
            float yy1 = std::max(input[i].bbox.ymin, input[j].bbox.ymin);
            float xx2 = std::min(input[i].bbox.xmax, input[j].bbox.xmax);
            float yy2 = std::min(input[i].bbox.ymax, input[j].bbox.ymax);

            if (xx2 > xx1 && yy2 > yy1) {
                float area_intersection = (xx2 - xx1) * (yy2 - yy1);
                float iou_ratio =
                        area_intersection / (input[j].area + input[i].area - area_intersection);;
                if (iou_ratio > iou_threshold) {
                    input[j].skip = 1;
                }
            }
        }
        result.push_back(input[i]);
    }
}

bool det_perception::find_person_area(std::vector<Detection> dect_src, int det_person_area){
    bool det_person_bool;
    for(int i=0;i<dect_src.size();i++){
        if(dect_src.at(i).id ==100){
            int w = dect_src.at(i).bbox.xmax - dect_src.at(i).bbox.xmin;
            int h = dect_src.at(i).bbox.ymax - dect_src.at(i).bbox.ymin;
            int area = w*h;
            if(area> det_person_area){
                return det_person_bool=true;
            }
        }
    }
    return false;
}


void det_perception::check_dist(float &area, float &min_dist, float &max_dist){
    if(area>0 && area<2500){
        min_dist=3;
    }else if(area>2500 && area<5000){
        min_dist=2;
        max_dist=3;
    }else if(area>28000){
        min_dist =1;
        max_dist =2;
    }else {
        min_dist=0;
        max_dist=0;
    }
}

void det_perception::det_side_persion_info(std::vector<Detection> &dect_src, int &person_num,
                                           float &person_max_score, float &person_max_area, float &person_min_dist, float &person_max_dist){
    person_num =dect_src.size();
    if(person_num==0) {
        cout << "det_side_info: " << "num:0, max_area:0, dist:0" << endl;
//        person_num=0;
//        person_score=0;
//        person_max_area=0;
//        person_min_dist=0;
//        person_max_dist=0;
    }else {
        for (int i = 0; i < dect_src.size(); i++) {
            if (dect_src.at(i).id == 0) {
                int w = dect_src.at(i).bbox.xmax - dect_src.at(i).bbox.xmin;
                int h = dect_src.at(i).bbox.ymax - dect_src.at(i).bbox.ymin;
                int area = w * h;
                if (area > person_max_area) {
                    person_max_area = area;
                    person_max_score = dect_src[i].score;
                }
            }
        }
    }
    check_dist(person_max_area, person_min_dist, person_max_dist);
}

void det_perception::task_release(){
    int res_ret = hbDNNReleaseTask(task_handle);
    if(res_ret!=0) cout<< "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void det_perception::perception_release() {
    for (int i = 0; i < input_count; i++) {
        int infree_ret = hbSysFreeMem(&(input_tensors[i].sysMem[0]));
        if(infree_ret!=0) cout << "hbSysFreeMem failed" << endl;
    }
    for (int i = 0; i < output_count; i++) {
        int outfree_ret = hbSysFreeMem(&(output_tensors[i].sysMem[0]));
        if(outfree_ret!=0) cout << "hbSysFreeMem failed" << endl;
    }
    int dnn_ret = hbDNNRelease(packed_dnn_handle);
    if(dnn_ret!=0) cout << "hbDNNRelease failed" << endl;

}

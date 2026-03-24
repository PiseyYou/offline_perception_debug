#include "cls_perception.h"

void cls_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor) {
//    int input_count = 0;
//    int output_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);

    /** Tips:
     * For input memory size:
     * *   input_memSize = input[i].properties.alignedByteSize
     * For output memory size:
     * *   output_memSize = output[i].properties.alignedByteSize
     */
    this->input = input_tensor;
    for (int i = 0; i < input_count; i++) {
        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);

        this->model_height = (input[i].properties).validShape.dimensionSize[2];
        this->model_width = (input[i].properties).validShape.dimensionSize[3];
//        cout << "model_height/model_width: " << model_height << "/" << model_width << endl;

        int input_memSize = input[i].properties.alignedByteSize;
        int input_allocCached_ret = hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
        input[i].properties.alignedShape = input[i].properties.validShape;

        if (inTensor_ret || input_allocCached_ret != 0)
            cout << "hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed" << endl;
    }

    this->output = output_tensor;
    for (int i = 0; i < output_count; i++) {
        int outTensor_ret = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int output_memSize = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);
        if (outTensor_ret || output_allocCached_ret != 0)
            cout << "hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed" << endl;
    }
//    return 0;
}


int cls_perception::read_image_2_tensor_as_nv12(Mat &bgr_mat,
                                    hbDNNTensor *input_tensor) {
    hbDNNTensor *input = input_tensor;
    hbDNNTensorProperties Properties = input->properties;
//    int tensor_id = 0;

    // NCHW , the struct of mobilenetv1_224x224 shape is NCHW
    int input_h = Properties.validShape.dimensionSize[2];
    int input_w = Properties.validShape.dimensionSize[3];

    // 转换为 YUV420 格式
    if (input_h % 2 || input_w % 2) {
//        VLOG(EXAMPLE_SYSTEM) << "Input img height and width must be aligned by 2!";
        cout << "Input img height and width must be aligned by 2!" << endl;
        return -1;
    }

    cv::Mat yuv_mat;
//    cv::cvtColor(cropped_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
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

int cls_perception::perception_postprocess_int64() {
    auto perception = std::shared_ptr<Perception>(new Perception);

    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    auto class_id = reinterpret_cast<int*>(output_tensors[0].sysMem[0].virAddr)[0];
    return class_id;
}


void cls_perception::get_topk_result(hbDNNTensor *tensor,
                     std::vector<Classification> &top_k_cls,
                     int top_k) {
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    std::priority_queue<Classification,
    std::vector<Classification>,
    std::greater<Classification>> queue;
    int *shape = tensor->properties.validShape.dimensionSize;
    // The type reinterpret_cast should be determined according to the output type
    // For example: HB_DNN_TENSOR_TYPE_F32 is float
    auto data = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
    auto shift = tensor->properties.shift.shiftData;
    auto scale = tensor->properties.scale.scaleData;
    int tensor_len = shape[0] * shape[1] * shape[2] * shape[3];
    // std::cout << "tensor_len: " << tensor_len << " shape0: " << shape[0] << " shape1: " << shape[1] << " shape2: " << shape[2]
    // << " shape3: " << shape[3] << std::endl;
    for (auto i = 0; i < tensor_len; i++) {
        float score = 0.0;
        if (tensor->properties.quantiType == SHIFT) {
            // std::cout << "shift" << std::endl;
            score = data[i] / (1 << shift[i]);
        } else if (tensor->properties.quantiType == SCALE) {
            // std::cout << "scale" << std::endl;
            score = data[i] * scale[i];
        } else {
            score = data[i];
        }
        queue.push(Classification(i, score, ""));
        if (queue.size() > top_k) {
            queue.pop();
        }
    }
    while (!queue.empty()) {
        top_k_cls.emplace_back(queue.top());
        queue.pop();
    }
    std::reverse(top_k_cls.begin(), top_k_cls.end());
}

void cls_perception::perception_init(const char *model_file_name){

    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    cout << "DNN runtime version: " << hbDNNGetVersion() << endl;

    {
        int inCount_ret = hbDNNGetInputCount(&input_count, dnn_handle);
        int outCount_ret = hbDNNGetOutputCount(&output_count, dnn_handle);
        if (inCount_ret || outCount_ret != 0) cout << "hbDhNNGetInputCount/hbDNNGetOutputCount failed" << endl;

        input_tensors.resize(input_count);
        output_tensors.resize(output_count);
//        prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle);
        prepare_tensor(input_tensors.data(), output_tensors.data());
    }
}


void cls_perception::perception_process(Mat &mat) {
    int nv12Ret = read_image_2_tensor_as_nv12(mat, input_tensors.data());
    if (nv12Ret != 0) cout << "read_image_2_tensor_as_nv12 failed" << endl;
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
    if (infer_ret != 0) cout << "hbDNNInfer failed" << endl;
    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    if (task_ret != 0) cout << "hbDNNWaitTaskDone failed" << endl;
};


int cls_perception::perception_postprocess(int top_k) {
    auto perception = std::shared_ptr<Perception>(new Perception);
    perception->type = Perception::CLS;

    std::vector<Classification> top_k_cls;
    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    get_topk_result(output, top_k_cls, top_k);
//    for (int i = 0; i < top_k; i++) {
//        cout << "TOP " << i << " result id: " << top_k_cls[i].id << endl;
//    }
    return top_k_cls.empty() ? -1 : top_k_cls[0].id;
}

int cls_perception::perception_process_bgr(Mat &bgr){
    perception_process(bgr);
    int ring_exsit = perception_postprocess(2);
    return ring_exsit;
}

void cls_perception::task_release() {
    int res_ret = hbDNNReleaseTask(task_handle);
    if (res_ret != 0) cout << "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void cls_perception::perception_release() {
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
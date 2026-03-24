#include <fstream>
#include"seg_perception.h"

void seg_perception::perception_init(const char *model_file_name) {

    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    int inCount_ret = hbDNNGetInputCount(&input_count, dnn_handle);
    int outCount_ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    if(inCount_ret || outCount_ret !=0) cout << "hbDNNGetInputCount/hbDNNGetOutputCount failed" << endl;

    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data());
}

int seg_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor) {
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);

    this->input = input_tensor;
    for (int i = 0; i < input_count; i++) {

        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
        int input_memSize = input[i].properties.alignedByteSize;
        int input_allocCached_ret = hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
        input[i].properties.alignedShape = input[i].properties.validShape;

        if(inTensor_ret || input_allocCached_ret !=0)
            cout << "hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed" << endl;
    }

    this->output = output_tensor;
    for (int i = 0; i < output_count; i++) {
        int outTensor_ret = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int output_memSize = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);

        if(outTensor_ret || output_allocCached_ret !=0)
            cout << "hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed" << endl;

    }
    return 0;
}

int seg_perception::prepare_mat_nv12(Mat originMat) {
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
    // LOG_INFO("w/h: {}, {}", mat.cols, mat.rows);
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
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return 0;

}

void seg_perception::perception_process(Mat &mat) {
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
    if(infer_ret!=0)
        cout << "hbDNNInfer failed" << endl;
    int task_ret = (task_handle, 0);

    if(task_ret!=0)
        cout << "hbDNNWaitTaskDone failed" << endl;
}

Mat seg_perception::perception_postprocess_int64(){
    hbDNNTensor *output_tensor = output_tensors.data();
    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }
    auto data = reinterpret_cast<int64_t *>(output_tensor->sysMem[0].virAddr);
    int *shape = output_tensor->properties.validShape.dimensionSize;
    int height = shape[2];
    int width = shape[3];
    int num_classes = shape[1];
    cv::Mat img_label(height, width, CV_8UC1);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int max_class = static_cast<int>(data[h * width + w]);
            img_label.at<uchar>(h, w) = static_cast<uchar>(max_class);
        }
    }
    return img_label;
}

string seg_perception::check_lab(Mat& lab_seg){
    string back_mark = "0";
    string lown_mark = "0";
    string road_mark = "0";
    string dyna_mark = "0";
    string obst_mark = "0";
    for (int i = 0; i < lab_seg.rows; ++i) {
        for (int j = 0; j < lab_seg.cols; ++j) {
            int value = lab_seg.at<uchar>(i, j);
//            if (value == 1 && back_mark == "0") {
//                back_mark = "1";
//            } else if (value == 2 && lown_mark == "0") {
            if (value == 2 && lown_mark == "0") {
                lown_mark = "1";
            } else if (value == 3 && road_mark == "0"){
                road_mark = "1";
            } else if(value == 4 && dyna_mark == "0"){
                dyna_mark = "1";
            }else if(value == 5 && obst_mark == "0"){
                obst_mark = "1";
            }else{
                continue;
            }
        }
    }
    string temp = back_mark + lown_mark + road_mark + dyna_mark + obst_mark;
    return temp;
}

int seg_perception::false_positive_suppress3(cv::Mat& lab_ori, cv::Mat& lab_dst, int threshold) {
    // 标签检查字符串
    std::string check_str = check_lab(lab_ori);
    lab_dst = lab_ori.clone();

    const int rows = lab_ori.rows;
    const int cols = lab_ori.cols;

    // 初始化类别掩膜和对应图像
    std::vector<cv::Mat> masks(4);  // 只处理类别2和3，因此只需要0~3
    std::vector<cv::Mat> mats(4);
    for (int i = 2; i <= 3; ++i) { // 仅初始化类别2和3
        mats[i] = cv::Mat::zeros(rows, cols, CV_8UC1);
    }

    // 提取每个类别的掩膜和对应图像
    for (int label = 2; label <= 3; ++label) { // 仅处理类别2和3
        if (check_str[label - 1] == '1') {
            cv::compare(lab_ori, label, masks[label], cv::CMP_EQ);
            lab_ori.copyTo(mats[label], masks[label]);
        }
    }

    // 根据threshold进行膨胀操作，并更新到lab_dst
    for (int label = 2; label <= 3; ++label) {
        if (check_str[label - 1] == '1') {
            int kernel_size = (label == 2 && threshold < 100) ? threshold :
                              (label == 3 && threshold >= 100 && threshold < 200) ? (threshold - 100) :
                              (threshold >= 200 && threshold < 300) ? (threshold - 200) : 0;
            if(kernel_size > 0){
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
                cv::dilate(mats[label], mats[label], kernel);
                lab_dst.setTo(label, mats[label]);
            }
        }
    }

    return true;
}

Mat seg_perception::perception_postprocess_int64_erode(Mat &img, int erode_pixel){
    perception_process(img);
    Mat result = perception_postprocess_int64();
    false_positive_suppress3(result, result, erode_pixel);
    task_release();
    return result;
}

bool seg_perception::check_person(Mat& lab_seg){
    for (int i = 0; i < lab_seg.rows; ++i) {
        for (int j = 0; j < lab_seg.cols; ++j) {
            int value = lab_seg.at<uchar>(i, j);
            if(value == 4){
                return true;
            }
        }
    }
    return false;
}

bool seg_perception::find_max_person_area(Mat &seg_lab, int area){
    cv::Mat side_dyna_mask;
    cv::Mat dyna_mat = cv::Mat::zeros(seg_lab.rows, seg_lab.cols, CV_8UC1);
    cv::compare(seg_lab, 4, side_dyna_mask, cv::CMP_EQ);
    seg_lab.copyTo(dyna_mat, side_dyna_mask);

    std::vector<std::vector<cv::Point>> contours;
    findContours(dyna_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    double maxArea = 0;

    if(contours.size()>1){
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
            }
        }
    }else{
        maxArea = cv::contourArea(contours[0]);
    }
    if(maxArea>area){
        return true;
    }
    return false;
}

void seg_perception::task_release(){
    int res_ret = hbDNNReleaseTask(task_handle);

    if(res_ret!=0)
        cout <<"hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void seg_perception::perception_release() {
    for (int i = 0; i < input_count; i++) {
        int infree_ret = hbSysFreeMem(&(input_tensors[i].sysMem[0]));

        if(infree_ret!=0)
            cout <<"hbSysFreeMem failed"<<endl;
    }
    for (int i = 0; i < output_count; i++) {
        int outfree_ret = hbSysFreeMem(&(output_tensors[i].sysMem[0]));

        if(outfree_ret!=0)
            cout <<"hbSysFreeMem failed"<<endl;
    }
    int dnn_ret = hbDNNRelease(packed_dnn_handle);

    if(dnn_ret!=0)
        cout <<"hbDNNRelease failed"<<endl;
}

#include "multi_sub_perception.h"
#include <cmath>

//std::vector<std::string> class_names = {"pole", "obstacle_tree", "fixed_obstacle", "cars", "static_obstacle", "dynamic_obstacle"};
std::vector<std::string> class_names = {"pole", "obstacle_tree", "fixed_obstacle", "cars", "static_obstacle", "dynamic_obstacle", "charge_station", "person"};

#define LIMIT_AREA 10

// ============== YOLOv5 解码相关常量 ==============
// 三个检测层的stride
const int STRIDES[] = {8, 16, 32};

// 三个检测层的特征图尺寸 (针对 640x384 输入)
const int FEAT_H[] = {48, 24, 12};  // 384/8=48, 384/16=24, 384/32=12
const int FEAT_W[] = {80, 40, 20};  // 640/8=80, 640/16=40, 640/32=20

// 每个尺度的 anchor 数量
const int NUM_ANCHORS_PER_SCALE[] = {
    FEAT_H[0] * FEAT_W[0] * 3,  // 48*80*3 = 11520
    FEAT_H[1] * FEAT_W[1] * 3,  // 24*40*3 = 2880
    FEAT_H[2] * FEAT_W[2] * 3   // 12*20*3 = 720
};

// Anchors for YOLOv5 (3个尺度, 每个尺度3个anchor)
const float ANCHORS[3][3][2] = {
    {{10, 13}, {16, 30}, {33, 23}},      // stride 8
    {{30, 61}, {62, 45}, {59, 119}},     // stride 16
    {{116, 90}, {156, 198}, {373, 326}}  // stride 32
};

// ============== 坐标解码函数 ==============
// 将模型输出的 sigmoid 值转换为实际像素坐标
void decode_box(int box_idx, float sigmoid_x, float sigmoid_y, float sigmoid_w, float sigmoid_h,
                float& center_x, float& center_y, float& width, float& height) {
    // 确定该 box 属于哪个尺度
    int scale = 0;
    int idx_in_scale = box_idx;

    if (box_idx < NUM_ANCHORS_PER_SCALE[0]) {
        scale = 0;
        idx_in_scale = box_idx;
    } else if (box_idx < NUM_ANCHORS_PER_SCALE[0] + NUM_ANCHORS_PER_SCALE[1]) {
        scale = 1;
        idx_in_scale = box_idx - NUM_ANCHORS_PER_SCALE[0];
    } else {
        scale = 2;
        idx_in_scale = box_idx - NUM_ANCHORS_PER_SCALE[0] - NUM_ANCHORS_PER_SCALE[1];
    }

    int feat_h = FEAT_H[scale];
    int feat_w = FEAT_W[scale];
    int stride = STRIDES[scale];

    // 计算 anchor index, grid_y, grid_x
    int anchor_idx = idx_in_scale / (feat_h * feat_w);
    int spatial_idx = idx_in_scale % (feat_h * feat_w);
    int grid_y = spatial_idx / feat_w;
    int grid_x = spatial_idx % feat_w;

    // 获取 anchor 尺寸
    float anchor_w = ANCHORS[scale][anchor_idx][0];
    float anchor_h = ANCHORS[scale][anchor_idx][1];

    // 坐标变换公式（与 PyTorch 训练代码一致）
    // xy: (sigmoid * 2 - 0.5 + grid) * stride
    // wh: (sigmoid * 2)^2 * anchor
    center_x = (sigmoid_x * 2.0f - 0.5f + grid_x) * stride;
    center_y = (sigmoid_y * 2.0f - 0.5f + grid_y) * stride;
    width = std::pow(sigmoid_w * 2.0f, 2) * anchor_w;
    height = std::pow(sigmoid_h * 2.0f, 2) * anchor_h;
}

void multi_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor) {
//    hbDNNGetInputCount(&input_count, dnn_handle);
//    hbDNNGetOutputCount(&output_count, dnn_handle);

    /** Tips:
     * For input memory size:
     * *   input_memSize = input[i].properties.alignedByteSize
     * For output memory size:
     * *   output_memSize = output[i].properties.alignedByteSize
     */
    this->input = input_tensor;
    for (int i = 0; i < input_count; i++) {
        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);

        model_height = (input[i].properties).validShape.dimensionSize[2];
        model_width = (input[i].properties).validShape.dimensionSize[3];
        cout << "model_height/model_width: " << model_height << "/" << model_width << endl;

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


int multi_perception::read_image_2_tensor_as_nv12(Mat &bgr_mat,
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

void multi_perception::perception_init(const char *model_file_name){

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


void multi_perception::perception_preprocess_bgr(Mat &mat) {
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


// 计算IoU（Intersection over Union）
float intersection_over_union(const Bbox& bbox1, const Bbox& bbox2) {
    float x1 = std::max(bbox1.xmin, bbox2.xmin);
    float y1 = std::max(bbox1.ymin, bbox2.ymin);
    float x2 = std::min(bbox1.xmax, bbox2.xmax);
    float y2 = std::min(bbox1.ymax, bbox2.ymax);

    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float bbox1_area = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin);
    float bbox2_area = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin);

    float union_area = bbox1_area + bbox2_area - intersection_area;

    return intersection_area / union_area;  // IoU公式
}


// NMS实现：非极大值抑制
std::vector<Detection> non_max_suppression(const std::vector<Detection>& detections,float iou_threshold, int max_det) {
    std::vector<Detection> result;

    // 按照置信度排序，置信度高的框排在前面
    std::vector<Detection> detections_copy = detections;
    std::sort(detections_copy.begin(), detections_copy.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    // NMS处理
    std::vector<bool> keep(detections_copy.size(), true);

    for (size_t i = 0; i < detections_copy.size(); ++i) {
        if (keep[i]) {
            const Detection& det_i = detections_copy[i];
            result.push_back(det_i);
            if (result.size() >= max_det) {
                break;
            }

            for (size_t j = i + 1; j < detections_copy.size(); ++j) {
                if (keep[j]) {
                    const Detection& det_j = detections_copy[j];
                    // 计算IoU (Intersection over Union)
                    float iou = intersection_over_union(det_i.bbox, det_j.bbox);

                    if (iou > iou_threshold) {
                        keep[j] = false;
                    }
                }
            }
        }
    }

    return result;
}


void multi_perception::perception_postprocess_no_argmax(std::vector<Detection> &detections, Mat &img_label) {
    auto perception = std::shared_ptr<Perception>(new Perception);
    perception->type = Perception::DET;

//    std::vector<Detection> detections;
    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    cv::Mat result_mat(ori_height, ori_width, CV_8UC1);
    result_mat = get_detect_result_no_argmax(output, 0.45, 0.45, detections, img_label);
//    imwrite("result_mat.png", result_mat);
}


void multi_perception::task_release() {
    int res_ret = hbDNNReleaseTask(task_handle);
    if (res_ret != 0) cout << "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void multi_perception::perception_release() {
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

int countCategory(const cv::Mat& mat, int category) {
    // 确保输入是单通道图像
    CV_Assert(mat.type() == CV_8UC1 || mat.type() == CV_32SC1 || mat.type() == CV_16UC1);

    // 使用 OpenCV 的按条件计数功能
    return cv::countNonZero(mat == category);
}

int multi_perception::false_positive_suppress3(cv::Mat& lab_ori, cv::Mat& lab_dst, int threshold) {
    lab_dst = lab_ori.clone();
    if(threshold==0) return 0;

    const int rows = lab_ori.rows;
    const int cols = lab_ori.cols;

    // 初始化类别2和类别3的掩膜和对应图像
    cv::Mat mask_2, mask_3;
    cv::Mat mat_2 = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat mat_3 = cv::Mat::zeros(rows, cols, CV_8UC1);

    // 提取类别2和类别3的掩膜和对应图像
    cv::compare(lab_ori, 2, mask_2, cv::CMP_EQ);
    lab_ori.copyTo(mat_2, mask_2);

    cv::compare(lab_ori, 3, mask_3, cv::CMP_EQ);
    lab_ori.copyTo(mat_3, mask_3);

    // 保持原始背景（除了类别2和3的区域）
    cv::Mat background = lab_ori.clone();
    // 将类别2和3的区域置为背景值1
    background.setTo(1, mask_2);
    background.setTo(1, mask_3);

    // 更新输出图像（先复制处理后的背景）
    lab_dst = background.clone();

    // 根据 threshold 进行膨胀操作，并更新到 lab_dst
    if (threshold > 2 && threshold < 100) {
        // 只对类别2进行膨胀
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(threshold, threshold));
        cv::dilate(mat_2, mat_2, kernel);
        lab_dst.setTo(2, mat_2);
    } else if (threshold >= 100 && threshold < 200) {
        // 只对类别3进行膨胀
        int road_threshold = threshold - 100;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(road_threshold, road_threshold));
        cv::dilate(mat_3, mat_3, kernel);
        lab_dst.setTo(3, mat_3);
    } else if (threshold >= 200 && threshold < 300) {
        // 对类别2和3都进行膨胀
        int ther = threshold - 200;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ther, ther));
        cv::dilate(mat_2, mat_2, kernel);
        cv::dilate(mat_3, mat_3, kernel);
        lab_dst.setTo(2, mat_2);
        lab_dst.setTo(3, mat_3);
    }

    return 0;
}

int multi_perception::false_positive_suppress4(cv::Mat& lab_ori, cv::Mat& lab_dst, int threshold) {
    lab_dst = lab_ori.clone();
    if(threshold==0) return 0;

    const int rows = lab_ori.rows;
    const int cols = lab_ori.cols;

    // 初始化类别2和类别3的掩膜和对应图像
    cv::Mat mask_2, mask_3;
    cv::Mat mat_2 = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat mat_3 = cv::Mat::zeros(rows, cols, CV_8UC1);

    // 提取类别2和类别3的掩膜和对应图像
    cv::compare(lab_ori, 2, mask_2, cv::CMP_EQ);
    lab_ori.copyTo(mat_2, mask_2);

    cv::compare(lab_ori, 3, mask_3, cv::CMP_EQ);
    lab_ori.copyTo(mat_3, mask_3);

    // 保持原始背景（除了类别2和3的区域）
    cv::Mat background = lab_ori.clone();
    // 将类别2和3的区域置为背景值1
    background.setTo(1, mask_2);
    background.setTo(1, mask_3);

    // 更新输出图像（先复制处理后的背景）
    lab_dst = background.clone();

    // 根据 threshold 进行膨胀操作，并更新到 lab_dst
    if (threshold > 2 && threshold < 100) {
        // 只对类别2进行膨胀
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(threshold, threshold));
        cv::dilate(mat_2, mat_2, kernel);

        // ========== 草坪边界平滑优化 ==========
        // 1. 使用椭圆形kernel进行闭运算，填充内部小孔洞
        cv::Mat smooth_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mat_2, mat_2, cv::MORPH_CLOSE, smooth_kernel);

        // 2. 轻微腐蚀后再膨胀，平滑边界锯齿
        cv::Mat boundary_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mat_2, mat_2, boundary_kernel);
        cv::dilate(mat_2, mat_2, boundary_kernel);
        // ========================================

        lab_dst.setTo(2, mat_2);
    } else if (threshold >= 100 && threshold < 200) {
        // 只对类别3进行膨胀
        int road_threshold = threshold - 100;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(road_threshold, road_threshold));
        cv::dilate(mat_3, mat_3, kernel);
        lab_dst.setTo(3, mat_3);
    } else if (threshold >= 200 && threshold < 300) {
        // 对类别2和3都进行膨胀
        int ther = threshold - 200;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ther, ther));
        cv::dilate(mat_2, mat_2, kernel);
        cv::dilate(mat_3, mat_3, kernel);

        // ========== 草坪边界平滑优化 ==========
        // 1. 使用椭圆形kernel进行闭运算，填充内部小孔洞
        cv::Mat smooth_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mat_2, mat_2, cv::MORPH_CLOSE, smooth_kernel);

        // 2. 轻微腐蚀后再膨胀，平滑边界锯齿
        cv::Mat boundary_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mat_2, mat_2, boundary_kernel);
        cv::dilate(mat_2, mat_2, boundary_kernel);
        // ========================================

        lab_dst.setTo(2, mat_2);
        lab_dst.setTo(3, mat_3);
    }

    return 0;
}


void multi_perception::perception_process_bgr_no_argmax_erode_mul(Mat &bgr, std::vector<Detection> &detections, Mat &img_label, int m_erode_pixel){
    perception_preprocess_bgr(bgr);
    perception_postprocess_no_argmax_mul(detections, img_label);
//    perception_postprocess_no_argmax_erode(detections, img_label, m_erode_pixel);
    false_positive_suppress4(img_label, img_label, m_erode_pixel);
    task_release();
}

void multi_perception::perception_postprocess_no_argmax_mul(std::vector<Detection> &detections, Mat &img_label) {
    auto perception = std::shared_ptr<Perception>(new Perception);
    perception->type = Perception::DET;

//    std::vector<Detection> detections;
    for (int i = 0; i < output_count; i++) {
        hbSysFlushMem(&(output_tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }

    cv::Mat result_mat(ori_height, ori_width, CV_8UC1);
    result_mat = get_detect_result_no_argmax_mul(output, 0.45, 0.45, detections, img_label);
//    imwrite("result_mat.png", result_mat);
}


Mat multi_perception::get_detect_result_no_argmax_mul(hbDNNTensor *output, float cls_thre, float iou_thre,
                                                  std::vector<Detection> &detections, cv::Mat &img_label) {
//    result.width = ori_width;
//    result.height = ori_height;

    auto data_detect = reinterpret_cast<float *>(output[0].sysMem[0].virAddr);
    int *shape = output[0].properties.validShape.dimensionSize;
    int num_boxes = shape[1];  // 25200: number of boxes
    int num_features = shape[2]; // 15: number of features (4 coordinates + 1 confidence + n classes)

//    std::vector<Detection> detections;
    // 计算缩放因子
    float scale_x = ori_width / float(model_width);  // 640 / 640, 宽度没有改变
    float scale_y = ori_height / float(model_height);  // 384 / 640, 高度改变了

    for (int i = 0; i < num_boxes; i++) {
        float confidence = data_detect[i * num_features + 4]; // confidence value

        // 如果置信度大于阈值，提取框和类别信息
        if (confidence > cls_thre) {
            // 假设数据是 center_x, center_y, width, height
            float center_x_f = static_cast<float>(data_detect[i * num_features + 0]);
            float center_y_f = static_cast<float>(data_detect[i * num_features + 1]);
            float width_f = static_cast<float>(data_detect[i * num_features + 2]);
            float height_f = static_cast<float>(data_detect[i * num_features + 3]);

            // 将中心坐标和宽高转换为左上角坐标和右下角坐标
            float xmin = center_x_f - width_f / 2;
            float ymin = center_y_f - height_f / 2;
            float xmax = center_x_f + width_f / 2;
            float ymax = (center_y_f + height_f / 2);

            // 调整框的位置和大小，基于原始图像尺寸 (640x384)
            xmin *= scale_x;
            ymin *= scale_y;
            xmax *= scale_x;
            ymax *= scale_y;

            int class_id = 0;
            float max_class_prob = 0.0f;

            // 获取具有最大概率的类别
            for (int c = 0; c < num_features - 5; c++) {
                float class_prob = data_detect[i * num_features + 5 + c];
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    class_id = c;
                }
            }
            std::string class_name = class_names[class_id];

            xmin = xmin>0 ? xmin:0;
            ymin = ymin>0 ? ymin:0;
            xmax = xmax<ori_width ? xmax:ori_width;
            ymax = ymax<ori_height ? ymax:ori_height;
//            if(xmin>370 && xmax<460 && ymin>355 && ymax<386) continue;
            if(xmin>355 && xmax<460 && ymin>355 && ymax<386) continue;
            // 使用计算出的坐标 (xmin, ymin, xmax, ymax) 构造 Bbox 和 Detection
            Bbox bbox(xmin, ymin, xmax, ymax);
            detections.push_back(Detection(class_id, confidence, bbox, class_name.c_str()));
        }
    }

    // 应用NMS
    detections = non_max_suppression(detections, iou_thre, 300);

    // 获取分割输出数据
//    auto data_seg = reinterpret_cast<float *>(output[4].sysMem[0].virAddr);
    auto *data_seg = reinterpret_cast<int64_t *>(output[4].sysMem[0].virAddr);
    shape = output[4].properties.validShape.dimensionSize;
    int height = shape[2];  // 640
    int width = shape[3];   // 640
    int num_classes = shape[1];  // 类别数

    cv::Mat segmentation_result(height, width, CV_8UC3);

//    uint8_t *img_label_ptr = img_label.ptr<uint8_t>();
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int max_class = static_cast<int>(data_seg[h * width + w]);

            // 使用缩放因子调整到原始图像的尺寸
            int orig_x = static_cast<int>(w * scale_x);  // 计算原图中的 x 坐标
            int orig_y = static_cast<int>(h * scale_y);  // 计算原图中的 y 坐标

            // 将类别信息存储到 mask_info 中
            img_label.at<uint8_t>(orig_y, orig_x) = static_cast<uint8_t>(max_class);
        }
    }

//    cv::Point topLeft(380, 360);
//    cv::Point bottomRight(465, 384);
//    cv::Rect roi(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
//    img_label(roi).setTo(cv::Scalar(2));
    return img_label;
}


void multi_perception::perception_process_bgr_no_argmax_erode(Mat &bgr, std::vector<Detection> &detections, Mat &img_label, Mat &lab_dst, int m_erode_pixel){
    perception_preprocess_bgr(bgr);
    perception_postprocess_no_argmax(detections, img_label);
    false_positive_suppress4(img_label, lab_dst, m_erode_pixel);
    task_release();
}



Mat multi_perception::get_detect_result_no_argmax(hbDNNTensor *output, float cls_thre, float iou_thre, std::vector<Detection> &detections, Mat &img_label) {
    auto data_detect = reinterpret_cast<float *>(output[0].sysMem[0].virAddr);
    int *shape = output[0].properties.validShape.dimensionSize;
    int num_boxes = shape[1];  // 25200: number of boxes
    int num_features = shape[2]; // 15: number of features (4 coordinates + 1 confidence + n classes)

//    std::vector<Detection> detections;
    // 计算缩放因子
    float scale_x = ori_width / float(model_width);  // 640 / 640, 宽度没有改变
    float scale_y = ori_height / float(model_height);  // 384 / 384, 高度改变了
    // std::cout << "scale_x: " << scale_x << ", scale_y: " << scale_y << std::endl;

    int detection_count = 0;
    for (int i = 0; i < num_boxes; i++) {
        float confidence = data_detect[i * num_features + 4]; // confidence value

        // 如果置信度大于阈值，提取框和类别信息
        if (confidence > cls_thre) {
            // 从模型输出获取 sigmoid 值（未经变换）
            float sigmoid_x = static_cast<float>(data_detect[i * num_features + 0]);
            float sigmoid_y = static_cast<float>(data_detect[i * num_features + 1]);
            float sigmoid_w = static_cast<float>(data_detect[i * num_features + 2]);
            float sigmoid_h = static_cast<float>(data_detect[i * num_features + 3]);

            // ============== 使用 decode_box 进行坐标解码 ==============
            float center_x_f, center_y_f, width_f, height_f;
            decode_box(i, sigmoid_x, sigmoid_y, sigmoid_w, sigmoid_h,
                       center_x_f, center_y_f, width_f, height_f);

            // 将中心坐标和宽高转换为左上角坐标和右下角坐标
            float xmin = center_x_f - width_f / 2;
            float ymin = center_y_f - height_f / 2;
            float xmax = center_x_f + width_f / 2;
            float ymax = center_y_f + height_f / 2;

            // 调整框的位置和大小，基于原始图像尺寸 (640x384)
            xmin *= scale_x;
            ymin *= scale_y;
            xmax *= scale_x;
            ymax *= scale_y;

            int class_id = 0;
            float max_class_prob = 0.0f;

            // 获取具有最大概率的类别
            for (int c = 0; c < num_features - 5; c++) {
                float class_prob = data_detect[i * num_features + 5 + c];
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    class_id = c;
                }
            }
//            cout << "class_id: " << class_id << endl;
            std::string class_name = class_names[class_id];

            xmin = xmin>0 ? xmin:0;
            ymin = ymin>0 ? ymin:0;
            xmax = xmax<ori_width ? xmax:ori_width;
            ymax = ymax<ori_height ? ymax:ori_height;

            // 使用计算出的坐标 (xmin, ymin, xmax, ymax) 构造 Bbox 和 Detection
            Bbox bbox(xmin, ymin, xmax, ymax);
            detections.push_back(Detection(class_id, confidence, bbox, class_name.c_str()));

            detection_count++;
        }
    }

    // std::cout << "\nTotal detections before NMS: " << detection_count << std::endl;

    // 应用NMS
    detections = non_max_suppression(detections, iou_thre, 40);
    // std::cout << "Total detections after NMS: " << detections.size() << std::endl;

    // 获取分割输出数据
    auto data_seg = reinterpret_cast<int32_t *>(output[1].sysMem[0].virAddr);
    shape = output[1].properties.validShape.dimensionSize;
    int height = shape[2];  // 384
    int width = shape[3];   // 640
    int num_classes = shape[1];  // 类别数

    cv::Mat segmentation_result(height, width, CV_8UC3);
    cv::Mat segmask_mat(height, width, CV_8UC1);

    // 遍历每个像素，找到类别分数最高的类别
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int max_class = static_cast<int>(data_seg[h * width + w]);

            // 使用缩放因子调整到原始图像的尺寸
            int orig_x = static_cast<int>(w * scale_x);  // 计算原图中的 x 坐标
            int orig_y = static_cast<int>(h * scale_y);  // 计算原图中的 y 坐标

            // 将类别信息存储到 mask_info 中
            img_label.at<uint8_t>(orig_y, orig_x) = static_cast<uint8_t>(max_class);
        }
    }

    return img_label;
}

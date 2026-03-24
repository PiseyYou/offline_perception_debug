#include "qr_cs_perception.h"

string qr_cs_perception::check_lab(Mat &lab_seg)
{
    string back_mark = "0";
    string lown_mark = "0";
    string road_mark = "0";
    string dyna_mark = "0";
    for (int i = 0; i < lab_seg.rows; ++i)
    {
        for (int j = 0; j < lab_seg.cols; ++j)
        {
            int value = lab_seg.at<uchar>(i, j);
            //            if (value == 1 && back_mark == "0") {
            //                back_mark = "1";
            //            } else if (value == 2 && lown_mark == "0") {
            if (value == 2 && lown_mark == "0")
            {
                lown_mark = "1";
            }
            else if (value == 3 && road_mark == "0")
            {
                road_mark = "1";
            }
            else if (value == 4 && dyna_mark == "0")
            {
                dyna_mark = "1";
            }
            else
            {
                continue;
            }
        }
    }
    string temp = back_mark + lown_mark + road_mark + dyna_mark;
    return temp;
}

int qr_cs_perception::countCategory(const cv::Mat &mat, int category)
{
    // 确保输入是单通道图像
    CV_Assert(mat.type() == CV_8UC1 || mat.type() == CV_32SC1 || mat.type() == CV_16UC1);

    // 使用 OpenCV 的按条件计数功能
    return cv::countNonZero(mat == category);
}

int qr_cs_perception::print_info(hbDNNTensorProperties properties)
{
    std::stringstream ss;
    std::string valid_shape = "( ";
    for (int k = 0; k < properties.validShape.numDimensions; k++)
    {
        valid_shape += std::to_string(properties.validShape.dimensionSize[k]);
        if (k != properties.validShape.numDimensions - 1)
        {
            valid_shape += ", ";
        }
    }
    valid_shape += " )";
    ss << "model[" << 0 << "] validShape: " << valid_shape;

    std::string aligne_shape = "( ";
    for (int k = 0; k < properties.alignedShape.numDimensions; k++)
    {
        aligne_shape +=
            std::to_string(properties.alignedShape.dimensionSize[k]);
        if (k != properties.alignedShape.numDimensions - 1)
        {
            aligne_shape += ", ";
        }
    }
    aligne_shape += " )";
    ss << ", alignedShape: " << aligne_shape;
    ss << ", tensorType: " << properties.tensorType;
    ss << ", tensorLayout: " << properties.tensorLayout << std::endl;
    //    cout << "ss.str: " << ss.str();
    cout << ss.str();
    return 1;
}

void qr_cs_perception::perception_init(const char *model_file_name)
{

    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    int inCount_ret = hbDNNGetInputCount(&input_count, dnn_handle);
    int outCount_ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    if (inCount_ret || outCount_ret != 0)
        cout << "hbDNNGetInputCount/hbDNNGetOutputCount failed" << endl;

    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data());
}

int qr_cs_perception::prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor)
{
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);

    this->input = input_tensor;
    std::stringstream ss;
    for (int i = 0; i < input_count; i++)
    {

        int inTensor_ret = hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
        //        cout << "model h/w: " << input[i].properties.validShape.dimensionSize[2] << "/" << input[i].properties.validShape.dimensionSize[3] << endl;
        int input_memSize = input[i].properties.alignedByteSize;
        int input_allocCached_ret = hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
        input[i].properties.alignedShape = input[i].properties.validShape;

        if (inTensor_ret || input_allocCached_ret != 0)
            cout << "hbDNNGetInputTensorProperties/input_hbSysAllocCachedMem failed" << endl;

        print_info(input[i].properties);
    }

    this->output = output_tensor;
    for (int i = 0; i < output_count; i++)
    {
        int outTensor_ret = hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int output_memSize = output[i].properties.alignedByteSize;
        int output_allocCached_ret = hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);
        if (outTensor_ret || output_allocCached_ret != 0)
            cout << "hbDNNGetOutputTensorProperties/output_hbSysAllocCachedMem failed" << endl;

        print_info(output[i].properties);
    }
    return 0;
}

int qr_cs_perception::prepare_mat_nv12(Mat originMat)
{
    //    auto start4 = std::chrono::high_resolution_clock::now();
    hbDNNTensor *input = input_tensors.data();
    hbDNNTensorProperties Properties = input->properties;
    int input_h = Properties.validShape.dimensionSize[1];
    int input_w = Properties.validShape.dimensionSize[2];
    if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        input_h = Properties.validShape.dimensionSize[2];
        input_w = Properties.validShape.dimensionSize[3];
    }

    // resize
    cv::Mat mat;
    mat.create(input_h, input_w, originMat.type());
    cv::resize(originMat, mat, mat.size(), 0, 0);
    //    cout << "w/h: " << mat.cols << " " << mat.rows << endl;
    // convert to YUV420
    if (input_h % 2 || input_w % 2)
    {
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

    for (int32_t i = 0; i < uv_width * uv_height; i++)
    {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    return 0;
};

void qr_cs_perception::perception_process(Mat &mat)
{
    int nv12Ret = prepare_mat_nv12(mat);
    for (int j = 0; j < input_count; j++)
    {
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
    if (infer_ret != 0)
        cout << "hbDNNInfer failed" << endl;
    int task_ret = hbDNNWaitTaskDone(task_handle, 0);
    if (task_ret != 0)
        cout << "hbDNNWaitTaskDone failed" << endl;
}

Mat qr_cs_perception::perception_postprocess_int64()
{
    hbDNNTensor *output_tensor = output_tensors.data();
    for (int i = 0; i < output_count; i++)
    {
        hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }
    auto data = reinterpret_cast<int64_t *>(output_tensor->sysMem[0].virAddr);
    int *shape = output_tensor->properties.validShape.dimensionSize;
    int height = shape[2];
    int width = shape[3];
    int num_classes = shape[1];
    cv::Mat img_label(height, width, CV_8UC1);
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            int max_class = static_cast<int>(data[h * width + w]);
            img_label.at<uchar>(h, w) = static_cast<uchar>(max_class);
        }
    }

    //    int rainy_mat_count = countCategory(img_label, 1);
    //    cout << "rainy_mat_count: " <<rainy_mat_count << endl;

    //    std::unordered_map<int, int> result = countLabels(img_label);
    //    // 输出结果
    //    std::cout << "Label counts:" << std::endl;
    //    for (const auto& pair : result) {
    //        std::cout << "Label " << pair.first << ": " << pair.second << " pixels" << std::endl;
    //    }
    return img_label;
}

Mat qr_cs_perception::perception_postprocess_yolov8n_int64()
{
    // 清空上次的结果
    decoded_bboxes_all_.clear();
    decoded_scores_all_.clear();
    // decoded_classes_all_.clear();
    decoded_mces_all_.clear();

    float conf_thres_raw = -log(1 / score_threshold - 1);
    hbSysFlushMem(&(output[order[9]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    std::vector<float> proto_data_dequant(H_4 * W_4 * mces);
    // auto begin_time = std::chrono::system_clock::now();

    auto *proto_data_float = reinterpret_cast<float *>(output[order[9]].sysMem[0].virAddr);
    // 复制数据
    for (int i = 0; i < H_4 * W_4 * mces; ++i)
    {
        proto_data_dequant[i] = proto_data_float[i];
    }
    // std::cout << "Using NONE quantization (float) for proto mask" << std::endl;

    // 创建存储掩码原型的矩阵
    cv::Mat proto_mat(H_4 * W_4, mces, CV_32F, proto_data_dequant.data());

    // std::cout << "\033[31m Proto processing time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
    // 处理三个尺度的输出
    ProcessFeatureMap(0, 1, 2, 8.0, conf_thres_raw, 's');
    ProcessFeatureMap(3, 4, 5, 16.0, conf_thres_raw, 'm');
    ProcessFeatureMap(6, 7, 8, 32.0, conf_thres_raw, 'l');

    // std::cout << "Feature map processing done. Total detections before NMS: " << decoded_bboxes_all_.size() << std::endl;
    // 对每个类别进行NMS
    // cv::Mat img_display = resized_img_.clone();
    // cv::Mat img_display_semantic = resized_img_.clone();
    // 创建掩膜图像
    // cv::Mat zeros = cv::Mat::zeros(input_h_, input_w_, CV_8UC3);
    // 创建最终合成图像（原图+掩膜）
    // cv::Mat result_overlay;

    // 执行NMS
    std::vector<cv::Rect2d> final_bboxes;
    std::vector<float> final_scores;
    // std::vector<int> final_class_ids;
    std::vector<std::vector<float>> final_mces;

    NmsProcess(final_bboxes, final_scores, final_mces);

    cv::Mat img_label = cv::Mat::zeros(height, width, CV_8UC1);

    // 找出分数最高的检测框，强制只出一个框
    int max_score_index = 0;
    float max_score = 0.0f;

    for (size_t i = 0; i < final_scores.size(); i++)
    {
        if (final_scores[i] > max_score)
        {
            max_score = final_scores[i];
            max_score_index = i;
        }
    }

    // 处理检测结果
    if (!final_bboxes.empty())
    {
        cv::Rect2d bbox = final_bboxes[max_score_index];
        // int cls_id = final_class_ids[max_score_index];
        float score = final_scores[max_score_index];
        std::vector<float> &mce = final_mces[max_score_index];

        // 确保类别ID在有效范围内
        // if ((cls_id - 100) < 0 || (cls_id - 100) >= static_cast<int>(CLASSES_LIST.size())) {
        //     std::cerr << "Warning: Invalid class ID: " << cls_id - 100 << ", defaulting to 0" << std::endl;
        //     cls_id = 0;
        // }

        // std::cout << "Object " << i+1 << ": " << CLASSES_LIST[cls_id - 100] << ", score="
        //         << std::fixed << std::setprecision(4) << score
        //         << ", bbox=(" << bbox.x << "," << bbox.y << ","
        //         << bbox.width << "," << bbox.height << ")" << std::endl;
        // PlotBboxs(img_display, bbox, cls_id - 100, score);

        // 生成实例掩码
        cv::Mat mce_mat(1, mces, CV_32F, mce.data());
        cv::Mat instance_mask_flat = proto_mat * mce_mat.t();
        cv::Mat instance_mask_low_res = instance_mask_flat.reshape(1, H_4);

        // 应用sigmoid激活函数
        cv::Mat sigmoid_mask;
        cv::exp(-instance_mask_low_res, sigmoid_mask);
        sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);

        // 上采样到输入图像尺寸
        cv::Mat resized_sigmoid_mask;
        cv::resize(sigmoid_mask, resized_sigmoid_mask, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

        // 二值化掩码
        cv::Mat binary_mask;
        cv::threshold(resized_sigmoid_mask, binary_mask, 0.5, 1.0, cv::THRESH_BINARY);

        // 确保掩码为8位格式，以便与bitwise_and操作兼容
        cv::Mat binary_mask_8u;
        binary_mask.convertTo(binary_mask_8u, CV_8U, 255);

        // 裁剪掩码到检测框区域
        float x1 = std::max(0.0, bbox.x);
        float y1 = std::max(0.0, bbox.y);
        float x2 = std::min(static_cast<double>(width), bbox.x + bbox.width);
        float y2 = std::min(static_cast<double>(height), bbox.y + bbox.height);

        int mask_h = static_cast<int>(y2 - y1);
        int mask_w = static_cast<int>(x2 - x1);

        // if (mask_h <= 0 || mask_w <= 0)
        //     continue;

        // // 确保ROI不超出图像边界
        // if (x1 + mask_w > input_w_ || y1 + mask_h > input_h_)
        //     continue;

        cv::Rect roi(static_cast<int>(x1), static_cast<int>(y1), mask_w, mask_h);
        cv::Mat roi_mask = binary_mask_8u(roi);

        // 对实例掩码映射至于语义类别
        cv::Mat roi_img_label = img_label(roi);
        // 仅在ROI区域内设置值
        roi_img_label.setTo(1, roi_mask == 255); // 1即为二维码所在的像素值

        // // 创建彩色掩码
        // cv::Mat color_mask = cv::Mat::zeros(roi.height, roi.width, CV_8UC3);
        // color_mask.setTo(cv::Scalar(INSTANCE_COLORS[(cls_id - 100) % INSTANCE_COLORS.size()][0],
        //                            INSTANCE_COLORS[(cls_id - 100) % INSTANCE_COLORS.size()][1],
        //                            INSTANCE_COLORS[(cls_id - 100) % INSTANCE_COLORS.size()][2]));

        // // 应用掩码到颜色
        // cv::Mat color_instance_mask;
        // cv::bitwise_and(color_mask, color_mask, color_instance_mask, roi_mask);

        // // 将掩码添加到零图像
        // cv::Mat zeros_roi = zeros(roi);
        // cv::addWeighted(zeros_roi, 1.0, color_instance_mask, 0.5, 0, zeros_roi);
    }
    return img_label;
}

void qr_cs_perception::ProcessFeatureMap(int id_1,
                                         int id_2,
                                         int id_3,
                                         float stride,
                                         float conf_thres_raw,
                                         char scale)
{

    int cls_idx = order[id_1];
    int box_idx = order[id_2];
    int mce_idx = order[id_3];
    // float s_stride = 8.0f;
    int32_t h_level;
    int32_t w_level;
    switch (scale)
    {
    case 's':
        h_level = H_8; // 示例值，根据你的模型设置
        w_level = W_8;
        break;
    case 'm':
        h_level = H_16;
        w_level = W_16;
        break;
    case 'l':
        h_level = H_32;
        w_level = W_32;
        break;
    }

    // 检查反量化类型
    // if (output_tensors_[cls_idx].properties.quantiType != NONE) {
    //     std::cerr << "Warning: Classification output should have SCALE quantization"
    //              << output_tensors_[cls_idx].properties.quantiType << std::endl;
    // }
    // if (output_tensors_[box_idx].properties.quantiType != SCALE) {
    //     std::cerr << "Warning: Bounding box output should have SCALE quantization "
    //              << output_tensors_[box_idx].properties.quantiType << std::endl;
    // }
    // if (output_tensors_[mce_idx].properties.quantiType != SCALE) {
    //     std::cerr << "Warning: Mask coefficient output should have SCALE quantization" << std::endl;
    // }
    // 刷新内存
    hbSysFlushMem(&output[cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[mce_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    // 获取数据指针和量化尺度
    auto *cls_raw = reinterpret_cast<float *>(output[cls_idx].sysMem[0].virAddr);
    auto *box_raw = reinterpret_cast<int32_t *>(output[box_idx].sysMem[0].virAddr);
    auto *box_scale = reinterpret_cast<float *>(output[box_idx].properties.scale.scaleData);
    auto *mce_raw = reinterpret_cast<int32_t *>(output[mce_idx].sysMem[0].virAddr);
    auto *mce_scale = reinterpret_cast<float *>(output[mce_idx].properties.scale.scaleData);

    // 遍历特征图的每个位置
    for (int h = 0; h < h_level; h++)
    {
        for (int w = 0; w < w_level; w++)
        {
            // 计算当前位置在内存中的偏移
            int offset = h * w_level + w;

            // 获取当前位置的特征向量
            float *cur_cls_raw = cls_raw + offset * num_classes;
            int32_t *cur_box_raw = box_raw + offset * (4 * reg);
            int32_t *cur_mce_raw = mce_raw + offset * mces;

            // 找到分数最大的类别
            int cls_id = 0;
            // for (int i = 1; i < num_classes; i++) {
            //     if (cur_cls_raw[i] > cur_cls_raw[cls_id]) {
            //         cls_id = i;
            //     }
            // }
            // 如果置信度低于阈值，跳过处理
            if (cur_cls_raw[cls_id] < conf_thres_raw)
            {
                continue;
            }
            // 计算Sigmoid激活后的置信度
            float score = 1.0f / (1.0f + std::exp(-cur_cls_raw[cls_id]));
            // 使用DFL解码边界框
            float ltrb[4] = {0.0f}; // left, top, right, bottom offsets

            for (int i = 0; i < 4; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < reg; j++)
                {
                    int index = reg * i + j;
                    float dfl = std::exp(float(cur_box_raw[index]) * box_scale[index]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }
            // 如果无效框，跳过处理
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                continue;
            }

            // 计算输入尺寸下的边界框坐标
            float x1 = (w + 0.5f - ltrb[0]) * stride;
            float y1 = (h + 0.5f - ltrb[1]) * stride;
            float x2 = (w + 0.5f + ltrb[2]) * stride;
            float y2 = (h + 0.5f + ltrb[3]) * stride;

            // 反量化掩码系数
            std::vector<float> mask_coeffs(mces);
            for (int i = 0; i < mces; i++)
            {
                mask_coeffs[i] = float(cur_mce_raw[i]) * mce_scale[i];
            }

            // 保存解码结果
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= width && y2 <= height)
            {
                decoded_bboxes_all_.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                decoded_scores_all_.push_back(score);
                // decoded_classes_all_.push_back(cls_id);
                decoded_mces_all_.push_back(mask_coeffs);
            }
        }
    }
}

void qr_cs_perception::NmsProcess(std::vector<cv::Rect2d> &final_bboxes,
                                  std::vector<float> &final_scores,
                                  std::vector<std::vector<float>> &final_mces)
{
    std::vector<int> nms_indices_output;
    std::vector<int> original_indices_map;
    if (!decoded_bboxes_all_.empty())
    {
        std::vector<cv::Rect> nms_bboxes_cv;
        std::vector<float> nms_scores_filtered;

        for (size_t idx = 0; idx < decoded_bboxes_all_.size(); ++idx)
        {
            const auto &box = decoded_bboxes_all_[idx];
            int x = std::max(0.0, box.x);
            int y = std::max(0.0, box.y);
            int w = std::max(1.0, box.width);
            int h = std::max(1.0, box.height);
            // 确保边界框不超出图像
            if (x + w > width)
                w = width - x;
            if (y + h > height)
                h = height - y;
            if (w <= 0 || h <= 0)
                continue;

            nms_bboxes_cv.push_back(cv::Rect(x, y, w, h));
            nms_scores_filtered.push_back(decoded_scores_all_[idx]);
            original_indices_map.push_back(idx);
        }

        if (!nms_bboxes_cv.empty())
        {
            cv::dnn::NMSBoxes(nms_bboxes_cv, nms_scores_filtered, score_threshold, nms_threshold, nms_indices_output);
            // std::cout << "NMS done. Detections after NMS: " << nms_indices_output.size() << std::endl;

            // 收集NMS后的结果
            for (int filtered_idx : nms_indices_output)
            {
                if (filtered_idx < 0 || filtered_idx >= static_cast<int>(original_indices_map.size()))
                {
                    std::cerr << "Error: Invalid index from NMSBoxes output" << std::endl;
                    continue;
                }

                int original_idx = original_indices_map[filtered_idx];
                final_bboxes.push_back(decoded_bboxes_all_[original_idx]);
                final_scores.push_back(decoded_scores_all_[original_idx]);
                // final_class_ids.push_back(decoded_classes_all_[original_idx]);
                final_mces.push_back(decoded_mces_all_[original_idx]);
            }
        }
        else
        {
            std::cout << "No valid boxes remaining before NMS." << std::endl;
        }
    }
    else
    {
        std::cout << "No detections before NMS." << std::endl;
    }
}

// cv::Point qr_cs_perception::find_max_area(Mat &seg_lab, int target_label){
cv::Point qr_cs_perception::find_mp_inqr(Mat &seg_lab, int target_label)
{
    cv::Mat charge_staion_mask;
    cv::Mat dyna_mat = cv::Mat::zeros(seg_lab.rows, seg_lab.cols, CV_8UC1);
    cv::compare(seg_lab, target_label, charge_staion_mask, cv::CMP_EQ); // 将label==2,设定为255
    seg_lab.copyTo(dyna_mat, charge_staion_mask);
    //    int count = countCategory(charge_staion_mask, 2);
    //    cout << "cout: " <<charge_staion_mask << endl;

    //    std::unordered_map<int, int> result = countLabels(dyna_mat);
    //    // 输出结果
    //    std::cout << "Label counts:" << std::endl;
    //    for (const auto& pair : result) {
    //        std::cout << "Label " << pair.first << ": " << pair.second << " pixels" << std::endl;
    //    }

    std::vector<std::vector<cv::Point>> contours;
    findContours(dyna_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    double maxArea = 0;
    int max_contour_indx = 0;

    if (contours.size() > 0)
    {
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                max_contour_indx = i;
            }
        }
    }
    else
    {
        //        continue;
        maxArea = cv::contourArea(contours[0]);
    }

    //    cout << "maxArea is: " << maxArea << endl;
    max_contours = contours.at(max_contour_indx);
    cv::Moments m = moments(max_contours, true);
    mp = cv::Point(m.m10 / m.m00, m.m01 / m.m00); // 质心
    //    cout << "mp: " << mp << endl;

    //    return maxArea;
    return mp;
}

float qr_cs_perception::measure_angle()
{
    return 0.017453293 * (mp.x - 320) / 640;
}

void qr_cs_perception::task_release()
{
    int res_ret = hbDNNReleaseTask(task_handle);
    if (res_ret != 0)
        cout << "hbDNNReleaseTask failed" << endl;
    task_handle = nullptr;
}

void qr_cs_perception::perception_release()
{
    for (int i = 0; i < input_count; i++)
    {
        int infree_ret = hbSysFreeMem(&(input_tensors[i].sysMem[0]));
        if (infree_ret != 0)
            cout << "hbSysFreeMem failed" << endl;
    }
    for (int i = 0; i < output_count; i++)
    {
        int outfree_ret = hbSysFreeMem(&(output_tensors[i].sysMem[0]));
        if (outfree_ret != 0)
            cout << "hbSysFreeMem failed" << endl;
    }
    int dnn_ret = hbDNNRelease(packed_dnn_handle);
    if (dnn_ret != 0)
        cout << "hbDNNRelease failed" << endl;
}
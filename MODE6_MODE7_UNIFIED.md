# Mode 6 & 7 统一裁剪逻辑 - 最终版本

## 更新日期
2026-04-13

## 统一的图像处理逻辑

Mode 6和Mode 7现在都使用**相同的裁剪逻辑**：

### 1. 图像裁剪（而非resize）
```cpp
// 从640x480裁剪成640x384（y从0到384）
cv::Rect cropRegion(0, 0, 640, 384);
cv::Mat croppedImg = left(cropRegion).clone();
```

### 2. 分割识别
- 输入：640x384的裁剪图片
- 输出：640x384的单通道分割图（label_map）

### 3. 深度图处理
```cpp
// 深度计算使用原始480高度的灰度图
Mat depth_cal = stereo_multi_match.stereo_multi_process(grayImageL, grayImageR, false);

// 裁剪深度图从480到384（y从0到384）
cv::Mat depth_384 = depth_cal(cv::Rect(0, 0, 640, 384)).clone();
```

### 4. 融合
- 640x384的单通道图片（label_map）
- 640x384的深度图（depth_384）
- 640x384的彩色图片（croppedImg）

### 5. 可视化拼接
```
横向拼接：
[原图640x384] [纯色分割640x384] [叠加图640x384] = 1920x384

纵向拼接：
[1920x384上方图片]
[点云可视化1920x960]
= 1920x1344总尺寸
```

## 关键代码修改

### process函数
```cpp
// Mode 6和7都使用原始640x480图片（在processMode内部裁剪）
if (config_.infer_mode == 6 || config_.infer_mode == 7)
{
  processed_left = left_img.clone();  // 保持640x480
}
else
{
  resize(left_img, processed_left, Size(640, 384));  // 其他模式resize
}
```

### processMode6函数（新逻辑）
```cpp
// 1. 裁剪（不是resize！）
cv::Rect cropRegion(0, 0, 640, 384);
cv::Mat croppedImg = left(cropRegion).clone();

// 2. 使用裁剪后的图片进行分割
mulSubPerception.perception_process_bgr_no_argmax_erode(
    croppedImg, detections, img_label, lab_out, config_.erode_pixel);

// 3. 裁剪深度图
cv::Mat depth_384 = depth_cal(cv::Rect(0, 0, 640, 384)).clone();

// 4. 融合
stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
    depth_384, result.label_map, dect_dst, croppedImg, ...);
```

### processMode7函数（保持一致）
```cpp
// 与Mode 6完全相同的逻辑
cv::Rect cropRegion(0, 0, 640, 384);
cv::Mat croppedImg = left(cropRegion).clone();
// ... 其余逻辑相同
```

## 验证结果

### Mode 6 (Sub-task)
✅ 正常运行
✅ 点云数量：3836、3727点
✅ 输出图片：1920x1344
✅ 处理速度：~2.4秒/帧
✅ 使用裁剪逻辑（不是resize）

### Mode 7 (DSG Night)
✅ 正常运行
✅ 点云数量：4588点
✅ 输出图片：1920x1344
✅ 处理速度：~2.4秒/帧
✅ 使用裁剪逻辑（不是resize）

## 测试输出示例

### Mode 6
```
[Mode 6] Sub-task recognition
[Step 1/3] Sub-task inference done: 2332.16 ms
[Step 2/3] Depth computation done: 9.40 ms
[Step 3/3] Fusion done: 0.64 ms
Total processing time: 2371.44 ms
Point cloud size: 3836 points
```

### Mode 7
```
[Mode 7] DSG Night recognition
[Step 2/3] Depth computation done: 8.90 ms
[Step 3/3] Fusion done: 0.57 ms
Total processing time: 2436.36 ms
Point cloud size: 4588 points
```

## 与之前的差异

### Mode 6 之前的逻辑（错误）
1. 裁剪到640x432
2. Resize到640x384进行推理
3. 还原到640x432
4. Pad到640x480进行深度处理
5. 裁剪回640x432进行融合
6. 输出1920x1392

### Mode 6 现在的逻辑（正确）
1. 裁剪到640x384
2. 使用640x384进行推理
3. 深度图裁剪到640x384
4. 融合使用640x384
5. 输出1920x1344

## 图像布局（两个模式相同）

```
┌─────────────────────────────────────────────────────────┐
│  原图(640x384)  │  纯色分割(640x384)  │  叠加图(640x384)  │  ← 1920x384
├─────────────────────────────────────────────────────────┤
│                                                         │
│              点云可视化 (1920x960)                        │  ← 1920x960
│         (label视图640x480 + RGB视图640x480)              │
│                                                         │
└─────────────────────────────────────────────────────────┘
                    总尺寸: 1920x1344
```

## 关键要点

1. **裁剪而非Resize**：从640x480直接裁剪到640x384，保留上部384行
2. **统一逻辑**：Mode 6和Mode 7使用完全相同的图像处理流程
3. **统一输出**：两个模式的输出图片尺寸都是1920x1344
4. **简化代码**：移除了复杂的resize和还原逻辑

所有修改已完成并测试通过！

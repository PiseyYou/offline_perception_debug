# 图像处理逻辑更新说明

## 更新日期
2026-04-13

## 修改内容

### 原逻辑（错误）
- 输入图片被resize到640x384
- 使用resize后的图片进行处理

### 新逻辑（正确）

#### 1. 图像裁剪（而非resize）
```cpp
// 从640x480裁剪成640x384（y从0到384）
cv::Rect cropRegion(0, 0, 640, 384);
cv::Mat croppedImg = left(cropRegion).clone();
```

#### 2. 分割识别
- 输入：640x384的裁剪图片
- 输出：640x384的单通道分割图（lab_dst）

#### 3. 深度图处理
```cpp
// 深度计算使用原始480高度的灰度图
Mat depth_cal = stereo_multi_match.stereo_multi_process(grayImageL, grayImageR, false);

// 裁剪深度图从480到384（y从0到384）
cv::Mat depth_384 = depth_cal(cv::Rect(0, 0, 640, 384)).clone();
```

#### 4. 融合
- 640x384的单通道图片（lab_dst）
- 640x384的深度图（depth_384）
- 640x384的彩色图片（croppedImg）

#### 5. 可视化拼接
```
横向拼接：
[原图640x384] [纯色分割640x384] [叠加图640x384] = 1920x384

纵向拼接：
[1920x384上方图片]
[点云可视化1920x960]
= 1920x1344总尺寸
```

## 关键修改点

### process函数
```cpp
// 对于mode 7，直接使用原始640x480图片（在processMode7内部裁剪）
if (config_.infer_mode == 7)
{
  processed_left = left_img.clone();  // 保持640x480
}
else
{
  resize(left_img, processed_left, Size(640, 384));  // 其他模式resize
}
```

### processMode7函数
```cpp
// 1. 裁剪（不是resize）
cv::Rect cropRegion(0, 0, 640, 384);
cv::Mat croppedImg = left(cropRegion).clone();

// 2. 使用裁剪后的图片进行分割
dsgPerception_.perception_process_bgr_no_argmax_erode(
    croppedImg, dect_src, img_label, lab_out, config_.erode_pixel);

// 3. 裁剪深度图
cv::Mat depth_384 = depth_cal(cv::Rect(0, 0, 640, 384)).clone();

// 4. 融合
stereo_multi_match.stereo_process_pci_depth_rgb_seg_det_fusion(
    depth_384, lab_dst, dect_dst, croppedImg, xyz_rgbi_cloud, out_xyz_rgbi_cloud);
```

## 验证结果

✅ 图片尺寸：1920x1344（正确）
✅ 点云数量：4302、4177等（正常）
✅ 处理速度：~2.2秒/帧
✅ 输出文件：图片和PCD都正确生成

## 输出示例

```
[Image 1/12] perception_stereo_165101_558_mapping_area_1769118661_MUL.jpg
Full image size: [1280 x 480] -> Left/Right: [640 x 480]

======= Processing Frame 0 =======
Mode: 7
[Mode 7] DSG Night recognition
[Step 2/3] Depth computation done: 8.53 ms
[Step 3/3] Fusion done: 0.53 ms
Total processing time: 2270.86 ms
Point cloud size: 4302 points
```

## 图像布局

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

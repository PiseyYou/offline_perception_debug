# Mode 6 修复说明

## 问题描述

运行Mode 6时出现以下错误：
```
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.6.0) error: (-215:Assertion failed) !fixedSize() || ((Mat*)obj)->size.operator()() == Size(_cols, _rows) in function 'create'
```

## 根本原因

在修改process函数以支持Mode 7的裁剪逻辑时，所有模式都被改为传入resize后的640x384图片。但是Mode 6的代码期望输入是640x480的图片，因为它内部需要：
1. 裁剪到640x432
2. Resize到640x384进行推理
3. 再还原到640x432

## 解决方案

修改process函数，让Mode 6和Mode 7都使用原始的640x480图片：

```cpp
// 对于mode 6和7，直接使用原始640x480图片（在processMode内部处理尺寸）
// 对于其他mode，resize到640x384
Mat processed_left, processed_right;
if (config_.infer_mode == 6 || config_.infer_mode == 7)
{
  // Mode 6 & 7: 使用原始640x480图片
  processed_left = left_img.clone();
  if (!right_img.empty())
  {
    processed_right = right_img.clone();
  }
}
else
{
  // 其他模式: Resize到640x384
  resize(left_img, processed_left, Size(640, 384));
  if (!right_img.empty())
  {
    resize(right_img, processed_right, Size(640, 384));
  }
}
```

## 各模式的图像处理逻辑

### Mode 6 (Sub-task)
- **输入**：640x480
- **处理**：
  1. 裁剪到640x432
  2. Resize到640x384进行推理
  3. 还原label到640x432
  4. Pad到640x480进行深度处理
  5. 裁剪回640x432进行融合
- **输出图片尺寸**：1920x1392（3张640x432横拼 + 点云可视化）

### Mode 7 (DSG Night)
- **输入**：640x480
- **处理**：
  1. 裁剪到640x384（y从0到384）
  2. 使用640x384进行推理
  3. 深度图也裁剪到640x384
  4. 融合使用640x384
- **输出图片尺寸**：1920x1344（3张640x384横拼 + 点云可视化）

### 其他模式 (0-5)
- **输入**：640x480
- **处理**：Resize到640x384
- **输出**：根据各模式的具体实现

## 验证结果

### Mode 6
✅ 正常运行
✅ 点云数量：4714、4250点
✅ 输出图片：1920x1392
✅ 处理速度：~2.2秒/帧

### Mode 7
✅ 正常运行
✅ 点云数量：4588点
✅ 输出图片：1920x1344
✅ 处理速度：~2.3秒/帧

## 测试输出示例

### Mode 6
```
[Mode 6] Sub-task recognition
[Step 1/3] Sub-task inference done: 2245.9 ms
[Step 2/4] Depth computation done: 8.16 ms
[Step 3/4] Depth inpainting done: 0.05 ms
[Step 4/4] Fusion done: 0.69 ms
Total processing time: 2284.74 ms
Point cloud size: 4714 points
```

### Mode 7
```
[Mode 7] DSG Night recognition
[Step 2/3] Depth computation done: 8.82 ms
[Step 3/3] Fusion done: 0.53 ms
Total processing time: 2313.62 ms
Point cloud size: 4588 points
```

## 关键修改文件

- `src/offline_perception_debug_384_sob.cpp`
  - process函数：添加Mode 6和7的特殊处理
  - processMode7函数：使用裁剪而非resize

所有修改已完成并测试通过！

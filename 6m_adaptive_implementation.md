# 夜间模式自适应立体匹配实现总结

## 实现目标
针对夜间双目匹配噪点问题，为 Mode 7 (DSG_Night) 实现自适应降噪策略：
- **自适应参数策略**：对低质量夜间帧使用更保守的参数（更大的blockSize和更严格的uniquenessRatio）来减少噪点
- **自然深度过滤**：在点云生成阶段通过深度值（< 6m）自然过滤，避免视差计算阶段的生硬截断

## 关键技术参数

### 相机参数
- fx = 245.1634049359
- fy = 245.1634049359  
- cx = 317.9649264254
- cy = 242.8759116226
- baseline ≈ 0.0799 meters (从投影矩阵计算得出)

### 深度过滤策略
- **不在视差计算阶段截断**：minDisparity = 0（保持完整范围）
- **在点云生成阶段自然过滤**：`if (d > 0 && d < 6.0f)` 
- 优点：避免生硬的截断效果，保持点云的自然过渡

### 图像质量评估阈值
- 低质量判定：brightness < 50.0 或 contrast < 20.0
- 使用 cv::meanStdDev 计算亮度（均值）和对比度（标准差）

### 自适应参数策略

#### StereoBM (主匹配器)
- **低质量图像**：
  - blockSize = 17 (vs 原始13)
  - uniquenessRatio = 10 (vs 原始4)
- **正常质量图像**：
  - blockSize = 13
  - uniquenessRatio = 4

#### StereoBM (半分辨率顶部)
- **低质量图像**：
  - blockSize = 17 (vs 原始13)
  - uniquenessRatio = 10 (保持)
- **正常质量图像**：
  - blockSize = 13
  - uniquenessRatio = 10

#### StereoSGBM (半分辨率底部)
- **低质量图像**：
  - blockSize = 9 (vs 原始5)
  - uniquenessRatio = 30 (vs 原始20)
- **正常质量图像**：
  - blockSize = 5
  - uniquenessRatio = 20

## 实现的新函数

### 1. 图像质量评估
```cpp
bool StereoMultiMatch::assess_image_quality(const cv::Mat& image, 
                                            double& brightness, 
                                            double& contrast)
```
- 使用 cv::meanStdDev 计算亮度和对比度
- 返回 true 表示高质量（brightness >= 50 且 contrast >= 20）

### 2. 自适应初始化函数
```cpp
void StereoMultiMatch::stereo_block_matcher_init_6m_adaptive(bool is_low_quality)
void StereoMultiMatch::half_top_stereo_block_matcher_init_6m_adaptive(bool is_low_quality)
void StereoMultiMatch::half_bottom_stereo_block_matcher_init_6m_adaptive(bool is_low_quality)
```
- 所有匹配器设置 minDisparity = 0 (保持完整范围，不生硬截断)
- 根据 is_low_quality 参数调整 blockSize 和 uniquenessRatio

### 3. 参数初始化
```cpp
void StereoMultiMatch::stereo_multi_param_init_6m_adaptive()
```
- 调用 stereo_dis_init() 和 stereo_base_param_init()
- 使用正常质量参数初始化匹配器（运行时动态调整）
- 设置 MultiScaleFilterParams

### 4. 自适应深度处理
```cpp
Mat StereoMultiMatch::stereo_multi_process_depth_6m_adaptive(Mat &rectifyL, Mat &rectifyR)
```
- 评估图像质量
- 根据质量重新初始化匹配器
- 执行与原始函数相同的处理流程

## 模式应用策略

### 使用自适应函数的模式
- **仅 Mode 7: DSG_Night (DSG夜间识别)**
  - 初始化时使用 `stereo_multi_param_init_6m_adaptive()`
  - 运行时 `stereo_multi_process()` 自动使用自适应参数
  - 点云生成时自然过滤 6m 以外的点

### 保持原始函数的模式
- **Mode 0-6: 所有其他模式**
  - Mode 0: Disable (无处理)
  - Mode 1: OnlyDepth (仅深度)
  - Mode 2: Detection (检测融合深度)
  - Mode 3: Segmentation (分割融合深度)
  - Mode 4: ArUco (简化版)
  - Mode 5: MultiTask (多任务识别)
  - Mode 6: SubTask (子任务识别)
  - **原因**：这些模式主要用于白天或正常光照条件，不需要夜间降噪优化

## 代码修改位置

### 头文件
- `/home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug/include/stereo_multi_match.h`
  - 添加2个公共方法声明
  - 添加4个私有方法声明

### 实现文件
- `/home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug/src/stereo_multi_match.cpp`
  - 在 line 103 后添加6个新函数实现
  - 所有初始化函数使用 minDisparity = 0（不截断）

### 主处理文件
- `/home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug/src/offline_perception_debug_432_sob.cpp`
  - 修改 init() 函数：仅 Mode 7 使用自适应初始化
  - Mode 0-6 保持使用原始初始化和处理函数

## 深度过滤实现

所有点云生成函数已经包含 6m 深度过滤：

### stereo_process_pc_rgbl_depth (Mode 1)
```cpp
if (d >= 0 && d < 6.0f)  // 自然过滤 6m 以外的点
```

### stereo_process_pci_depth_rgb_seg_fusion (Mode 3)
```cpp
if (d > 0 && d < 6.0f)  // 自然过滤
```

### stereo_process_pci_depth_rgb_seg_det_fusion (Mode 7)
```cpp
if (d > 0 && d < 6.0f)  // 自然过滤
```

## 预期效果

### 噪点抑制
- 低质量图像使用更大的blockSize（17 vs 13），增强匹配稳定性
- 更严格的uniquenessRatio（10 vs 4），过滤不可靠匹配
- 保持检测能力：不跳过帧，所有帧都进行处理

### 自然过渡
- 视差计算保持完整范围（minDisparity = 0）
- 点云生成阶段通过深度值自然过滤
- 避免生硬的截断效果，点云边界更自然

### 自适应性
- 运行时动态评估图像质量
- 根据质量自动调整参数
- 对正常质量图像保持原有性能

### 专注夜间场景
- 仅 Mode 7 (DSG_Night) 使用自适应策略
- 其他模式保持原有性能和行为
- 针对性优化，不影响白天场景

## 编译状态
✅ 编译成功
- 所有目标构建完成
- 无编译错误
- 可执行文件生成：`./build/offline_perception_debug_432_sob`


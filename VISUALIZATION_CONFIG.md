# 可视化配置优化说明

## 更新日期
2026-04-13

## 问题描述

### 问题1：底部被强制置为label==2
分割图底部（y=370到384）被人为强制设置为label==2（草地），导致底部区域的真实分割结果被覆盖。

### 问题2：检测框显示硬编码
检测框的显示被硬编码为`false`，无法灵活配置是否显示检测框、类别和置信度。

## 解决方案

### 1. 添加配置选项

在`Config`结构体中添加两个新的配置选项：

```cpp
struct Config
{
    // ... 其他配置 ...
    
    // 可视化配置
    bool enable_draw_detection_box = true;   // 是否在可视化中显示检测框
    bool enable_force_bottom_label = false;  // 是否强制底部区域为label==2（草地）
};
```

### 2. 修改filterLabelDect函数

**函数签名修改：**
```cpp
void filterLabelDect(cv::Mat &src_lab, std::vector<Detection> &dect_src,
                     cv::Mat &lab_dst, std::vector<Detection> &dect_dst,
                     bool enable_det = true, bool enable_force_bottom = false);
```

**实现修改：**
```cpp
// 可配置：是否强制底部区域为label==2（草地）
if (enable_force_bottom)
{
  int shift_high = 370;
  cv::Rect force_region(0, shift_high, 640, 384 - shift_high);
  lab_dst(force_region).setTo(cv::Scalar(2));
}
```

### 3. 修改可视化代码

**Mode 6和Mode 7的可视化：**
```cpp
// 使用配置选项控制是否显示检测框
pure_seg = drawResultOptimized(croppedImgVis, labVis, dect_dst, 
                               img_seg_show, config_.enable_draw_detection_box);
```

### 4. 修改函数调用

**Mode 6：**
```cpp
filterLabelDect(lab_out, detections, dst_label, dect_dst, 
                true, config_.enable_force_bottom_label);
```

**Mode 5：**
```cpp
filterLabelDect(img_label, detections, dst_label, dect_dst, 
                true, config_.enable_force_bottom_label);
```

## 配置说明

### enable_draw_detection_box（默认：true）
- **true**：在可视化图片中显示检测框、类别ID和置信度
- **false**：不显示检测框，只显示分割结果

### enable_force_bottom_label（默认：false）
- **true**：强制将底部区域（y=370到384）设置为label==2（草地）
- **false**：保留底部区域的真实分割结果

## 使用示例

### 示例1：显示检测框，不强制底部
```cpp
config.enable_draw_detection_box = true;   // 显示检测框
config.enable_force_bottom_label = false;  // 不强制底部
```

### 示例2：不显示检测框，强制底部为草地
```cpp
config.enable_draw_detection_box = false;  // 不显示检测框
config.enable_force_bottom_label = true;   // 强制底部为草地
```

## 配置输出

程序启动时会显示配置信息：
```
========== Configuration ==========
Inference mode: 6
Erode pixel: 205
Area threshold: 0.5
Detection threshold: 0.3
Draw detection box: enabled
Force bottom label: disabled
Red brick refine: disabled
Depth inpainting strategy: 0
===================================
```

## 检测框显示格式

当`enable_draw_detection_box = true`时，检测框会显示：
- **矩形框**：标识检测到的对象
- **类别ID**：显示对象的类别编号
- **置信度**：显示检测的置信度分数

## 影响的模式

这些配置选项影响以下模式：
- **Mode 5**：Multi-task模式
- **Mode 6**：Sub-task模式
- **Mode 7**：DSG Night模式

## 验证结果

✅ 配置选项正常工作
✅ 底部区域可以保留真实分割结果
✅ 检测框显示可以灵活配置
✅ 配置信息正确显示

## 关键文件修改

1. `src/offline_perception_debug_384_sob.cpp`
   - 添加配置选项
   - 修改可视化代码
   - 修改函数调用
   - 添加配置输出

2. `include/offline_utils.hpp`
   - 修改`filterLabelDect`函数签名

3. `src/offline_utils.cpp`
   - 修改`filterLabelDect`函数实现
   - 添加`enable_force_bottom`参数控制

## 注意事项

1. **默认行为变更**：
   - 检测框显示：从`false`改为`true`（默认显示）
   - 底部强制：从`true`改为`false`（默认不强制）

2. **向后兼容**：
   - `filterLabelDect`函数的新参数有默认值，保持向后兼容

3. **灵活配置**：
   - 可以在main函数中轻松修改配置
   - 未来可以从配置文件读取

所有修改已完成并测试通过！

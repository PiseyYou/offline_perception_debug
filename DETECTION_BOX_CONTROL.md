# 检测框融合控制优化

## 更新日期
2026-04-13

## 问题描述

当 `enable_draw_detection_box = false` 时，虽然可视化图片不显示检测框，但检测框的 label>=100 仍然被映射到分割结果的单通道图片上，影响了纯分割结果的准确性。

用户期望：
- `enable_draw_detection_box = true`：融合检测框到分割结果，显示检测框和类别名称
- `enable_draw_detection_box = false`：**完全不处理检测框**，只使用纯分割结果，不将 label>=100 映射到分割图上

## 根本原因

之前的实现中，`enable_draw_detection_box` 只控制可视化时是否绘制检测框，但 `filterLabelDect` 函数总是会被调用，将检测框区域的 label 修改为 id+100（例如 103, 104, 106, 107）。

## 解决方案

### 核心思路

根据 `enable_draw_detection_box` 配置，决定是否调用 `filterLabelDect` 函数：
- **true**：调用 `filterLabelDect`，融合检测框到分割结果
- **false**：跳过 `filterLabelDect`，直接使用纯分割结果

### Mode 6 (Sub-task) 修改

在 [offline_perception_debug_384_sob.cpp:1526](src/offline_perception_debug_384_sob.cpp#L1526) 修改：

```cpp
// 根据配置决定是否融合检测框到分割结果
if (config_.enable_draw_detection_box)
{
  // 融合检测框：将检测框区域的label映射为id+100
  filterLabelDect(lab_out, detections, dst_label, dect_dst, true, config_.enable_force_bottom_label);
}
else
{
  // 不融合检测框：只使用纯分割结果
  lab_out.copyTo(dst_label);

  // 可选：是否强制底部区域为label==2（草地）
  if (config_.enable_force_bottom_label)
  {
    int shift_high = 370;
    cv::Rect force_region(0, shift_high, 640, 384 - shift_high);
    dst_label(force_region).setTo(cv::Scalar(2));
  }

  // 处理label==0的像素，设置为2（草地）
  cv::Mat mask_zero;
  cv::compare(dst_label, 0, mask_zero, cv::CMP_EQ);
  dst_label.setTo(2, mask_zero);
}
```

### Mode 5 (Multi-task) 修改

在 [offline_perception_debug_384_sob.cpp:1244](src/offline_perception_debug_384_sob.cpp#L1244) 应用相同的逻辑。

### Mode 7 (DSG Night) 修改

Mode 7 不使用 `filterLabelDect`，而是直接映射检测框ID。在 [offline_perception_debug_384_sob.cpp:1418](src/offline_perception_debug_384_sob.cpp#L1418) 修改：

```cpp
// 根据配置决定是否处理检测框
if (config_.enable_draw_detection_box)
{
  // 融合检测框：将检测框ID映射为id+100
  for (const auto& det : dect_src) {
      Detection fixed_det = det;
      fixed_det.id += 100; // Map to 100+ format
      dect_dst.push_back(fixed_det);
  }
}
// 如果不显示检测框，dect_dst 保持为空
```

## 效果对比

### enable_draw_detection_box = true（融合检测框）

**分割结果**：
- 检测框区域的 label 被映射为 id+100
- 例如：障碍物检测框区域 label=104

**可视化**：
- 显示检测框矩形
- 显示类别名称（例如 "stat:0.88"）

**点云**：
- 包含 label>=100 的点（例如 label=104 的障碍物点）

**日志输出**：
```
========== 障碍物采样诊断 ==========
Label 104:
  - 标签图像素数: 5510
  - 有效深度像素: 0 (0%)
  - 采样点云数: 0 (-nan%)
========================================
```

### enable_draw_detection_box = false（纯分割结果）

**分割结果**：
- 只包含原始分割类别（0-12）
- 不包含 label>=100 的检测框映射

**可视化**：
- 不显示检测框矩形
- 不显示类别名称

**点云**：
- 只包含原始分割类别的点
- 不包含 label>=100 的点

**日志输出**：
```
========== 障碍物采样诊断 ==========
[DepthFilter] 深度一致性过滤剔除点数: 0
========================================
```

注意：没有 "Label 104" 的诊断信息，说明检测框没有被融合。

## 配置说明

在 Config 结构体中设置（行191）：

```cpp
// 可视化配置
bool enable_draw_detection_box = true;   // 是否融合检测框并显示
bool enable_force_bottom_label = false;  // 是否强制底部区域为label==2（草地）
```

- **true**：融合检测框到分割结果，显示检测框和类别名称
- **false**：完全不处理检测框，只使用纯分割结果

## 影响的模式

这个优化影响所有模式：
- ✅ Mode 5 (Multi-task)
- ✅ Mode 6 (Sub-task)
- ✅ Mode 7 (DSG Night)

## 验证结果

✅ `enable_draw_detection_box = true` 时，检测框融合到分割结果，显示类别名称  
✅ `enable_draw_detection_box = false` 时，完全不处理检测框，只使用纯分割结果  
✅ 单通道分割图不包含 label>=100 的检测框映射  
✅ 点云不包含 label>=100 的点  
✅ 可视化图片不显示检测框  
✅ 所有模式正常工作

## 关键文件修改

1. **src/offline_perception_debug_384_sob.cpp**
   - Mode 5：行1241-1268，根据配置决定是否调用 `filterLabelDect`
   - Mode 6：行1520-1550，根据配置决定是否调用 `filterLabelDect`
   - Mode 7：行1416-1428，根据配置决定是否映射检测框ID

## 技术细节

### filterLabelDect 函数的作用

`filterLabelDect` 函数会：
1. 将检测框区域的 label 修改为 id+100（例如 103, 104, 106, 107）
2. 将检测框信息添加到 `dect_dst` 列表
3. 应用特殊的过滤逻辑（例如障碍物检测框的背景/可行走区域判断）

### 跳过 filterLabelDect 的处理

当 `enable_draw_detection_box = false` 时，我们需要手动处理：
1. 复制原始分割结果：`lab_out.copyTo(dst_label)`
2. 可选的底部强制：根据 `enable_force_bottom_label` 配置
3. 处理 label==0 的像素：设置为 2（草地）

这样可以保证分割结果的纯净性，不受检测框影响。

所有修改已完成并测试通过！

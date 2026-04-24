# 检测框ID映射修复说明

## 更新日期
2026-04-13

## 问题描述

检测框在显示时，ID没有正确映射到`id+100`格式。虽然在label图中检测框区域被设置为`id+100`，但是传递给可视化函数的检测框列表中，ID仍然是原始值（例如3、4、6），导致显示的类别与实际label不一致。

## 问题分析

### 原始逻辑（错误）

在`filterLabelDect`函数中：
```cpp
// 设置label图中的值为id+100
roi_dst.setTo(target_id + 100);

// 但是添加到dect_dst时，ID没有改变
dect_dst.push_back(dect_src[i]);  // ❌ ID还是原始值
```

这导致：
- **Label图**：检测框区域的label是`id+100`（例如103、104、106）
- **检测框列表**：检测框的ID还是原始值（例如3、4、6）
- **可视化结果**：显示的类别与label图不一致

### 正确逻辑

检测框添加到`dect_dst`时，也需要将ID映射为`id+100`：
```cpp
// 设置label图中的值为id+100
roi_dst.setTo(target_id + 100);

// 创建新的检测框，ID也映射为id+100
Detection det_with_mapped_id = dect_src[i];
det_with_mapped_id.id = target_id + 100;  // ✅ ID映射为id+100
dect_dst.push_back(det_with_mapped_id);
```

## 修复方案

### 修改filterLabelDect函数

在`src/offline_utils.cpp`中修改所有添加检测框到`dect_dst`的地方：

#### 1. 处理id=3（车辆）
```cpp
if (target_id == 3)
{
  process_roi(103);
  Detection det_with_mapped_id = dect_src[i];
  det_with_mapped_id.id = 103;  // 映射为103
  dect_dst.push_back(det_with_mapped_id);
}
```

#### 2. 处理id=6（充电桩）
```cpp
else if (target_id == 6)
{
  process_roi(106);
  Detection det_with_mapped_id = dect_src[i];
  det_with_mapped_id.id = 106;  // 映射为106
  dect_dst.push_back(det_with_mapped_id);
}
```

#### 3. 处理id=7（人）
```cpp
else if (target_id == 7)
{
  cout << "[person] have been dect....." << endl;
  Detection det_with_mapped_id = dect_src[i];
  det_with_mapped_id.id = 107;  // 映射为107
  dect_dst.push_back(det_with_mapped_id);
}
```

#### 4. 处理id=4（障碍物）
```cpp
else if (target_id == 4)
{
  // ... 判断逻辑 ...
  
  // 情况2和3：添加检测框时映射ID
  roi_dst.setTo(104);
  Detection det_with_mapped_id = dect_src[i];
  det_with_mapped_id.id = 104;  // 映射为104
  dect_dst.push_back(det_with_mapped_id);
}
```

#### 5. 处理其他ID
```cpp
else
{
  if (enable_det)
  {
    roi_dst.setTo(target_id + 100);
    Detection det_with_mapped_id = dect_src[i];
    det_with_mapped_id.id = target_id + 100;  // 映射为id+100
    dect_dst.push_back(det_with_mapped_id);
  }
}
```

## ID映射表

| 原始ID | 类别 | 映射后ID | Label图值 |
|--------|------|----------|-----------|
| 3 | 车辆 | 103 | 103 |
| 4 | 障碍物 | 104 | 104 |
| 6 | 充电桩 | 106 | 106 |
| 7 | 人 | 107 | 107 |
| 其他 | - | id+100 | id+100 |

## 可视化效果

修复后，当`enable_draw_detection_box = true`时：
- **检测框**：显示矩形框标识对象位置
- **类别ID**：显示映射后的ID（103、104、106、107等）
- **置信度**：显示检测的置信度分数
- **一致性**：检测框显示的ID与label图中的值完全一致

## 影响的模式

这个修复影响所有使用`filterLabelDect`函数的模式：
- ✅ Mode 5 (Multi-task)
- ✅ Mode 6 (Sub-task)

注意：Mode 7 (DSG Night)不使用`filterLabelDect`，它在processMode7中直接处理ID映射。

## 验证结果

✅ 检测框ID正确映射为id+100
✅ Label图与检测框ID一致
✅ 可视化显示正确的类别ID
✅ 程序正常运行，点云数量正常

## 关键代码修改

**文件**：`src/offline_utils.cpp`
**函数**：`filterLabelDect`
**修改**：所有添加检测框到`dect_dst`的地方，都创建新的Detection对象并映射ID

## 示例

### 修复前
```
检测到车辆：
- Label图：区域值为103
- 检测框ID：3
- 显示：类别3（错误！）
```

### 修复后
```
检测到车辆：
- Label图：区域值为103
- 检测框ID：103
- 显示：类别103（正确！）
```

所有修改已完成并测试通过！

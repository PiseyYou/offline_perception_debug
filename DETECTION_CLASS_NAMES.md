# 检测框类别名称显示优化

## 更新日期
2026-04-13

## 问题描述

检测框标签显示的是"id_XXX"格式（例如"id_103:0.95"），不够直观。需要改为显示类别名称（例如"car:0.95"）。

## 根本原因

经过仔细排查，发现了两个关键问题：

1. **类别映射表未初始化**：`initMulClassMap()` 函数定义了但从未被调用，导致 `mul_map_class` 是空的
2. **配置被硬编码覆盖**：main 函数中有 `config.enable_draw_detection_box = true` 硬编码覆盖了 Config 结构体的默认值

## 解决方案

### 1. 在 main 函数中调用初始化函数

**关键修复**：在 main 函数开始时调用 `initMulClassMap()`：

```cpp
int main(int argc, char **argv)
{
  (void)argc; (void)argv;
  cout << "==================================================" << endl;
  cout << "  Offline Perception Debug Tool (Mode-Based)    " << endl;
  cout << "==================================================" << endl;

  // 初始化类别映射表 ⚠️ 必须在使用前调用！
  initMulClassMap();

  // 读取配置
  OfflinePerceptionProcessor::Config config;
  // ...
}
```

### 2. 修改函数签名

在 `include/offline_utils.hpp` 中，为 `drawResultOptimized()` 函数添加类别映射参数：

```cpp
cv::Mat drawResultOptimized(cv::Mat &img_src, cv::Mat &img_lab,
                            std::vector<Detection> &dect_src,
                            cv::Mat &img_seg_show, bool enable_draw_box = false,
                            const std::map<int, std::string> *class_map = nullptr);
```

### 3. 修改实现逻辑

在 `src/offline_utils.cpp` 中，修改标签生成逻辑：

```cpp
// 使用类别名称（如果提供了class_map）
std::string class_name;
if (class_map && class_map->count(det.id)) {
    class_name = class_map->at(det.id);
} else {
    class_name = "id_" + std::to_string(det.id);
}
std::string label = class_name + ":" + cv::format("%.2f", det.score);
```

### 4. 更新函数调用

在 `src/offline_perception_debug_384_sob.cpp` 中，所有调用 `drawResultOptimized()` 的地方都传入 `&mul_map_class`：

```cpp
// Mode 5
drawResultOptimized(croppedImg, label_432, dect_dst, img_seg_show, false, &mul_map_class);

// Mode 6 和 Mode 7
pure_seg = drawResultOptimized(croppedImgVis, labVis, dect_dst, img_seg_show, 
                               config_.enable_draw_detection_box, &mul_map_class);
```

### 5. 修复配置覆盖问题

在 main 函数中，注释掉硬编码的配置覆盖，让程序使用 Config 结构体中的默认值：

```cpp
// 可视化配置（使用 Config 结构体中的默认值）
// config.enable_draw_detection_box = true;   // 显示检测框、类别和置信度
// config.enable_force_bottom_label = false;  // 不强制底部为草地
```

这样可以通过修改 Config 结构体中的默认值来控制检测框显示。

## 类别映射表

根据 `initMulClassMap()` 函数定义的映射关系：

| ID | 类别名称 | 说明 |
|----|---------|------|
| 103 | car | 车辆 |
| 104 | stat | 静态障碍物 |
| 106 | chst | 充电桩 |
| 107 | pers | 人 |
| 101 | obst | 障碍物 |
| 102 | fixo | 固定障碍物 |
| 105 | dyna | 动态障碍物 |

## 显示效果

### 修改前
```
检测框标签：id_103:0.95
检测框标签：id_104:0.88
```

### 修改后
```
检测框标签：car:0.95
检测框标签：stat:0.88
```

## 配置控制

### 显示检测框（默认）
在 Config 结构体中设置：
```cpp
bool enable_draw_detection_box = true;   // 显示检测框
```

配置输出：
```
Draw detection box: enabled
```

### 隐藏检测框
在 Config 结构体中设置：
```cpp
bool enable_draw_detection_box = false;  // 不显示检测框
```

配置输出：
```
Draw detection box: disabled
```

当检测框显示关闭时，可视化图片只显示分割结果，不显示检测框、类别和置信度。

## 向后兼容

- 新参数 `class_map` 有默认值 `nullptr`，保持向后兼容
- 如果不提供类别映射，会回退到 "id_XXX" 格式
- 如果提供的映射中没有对应的ID，也会回退到 "id_XXX" 格式

## 影响的模式

这个优化影响所有使用 `drawResultOptimized()` 的模式：
- ✅ Mode 5 (Multi-task)
- ✅ Mode 6 (Sub-task)
- ✅ Mode 7 (DSG Night)

## 验证结果

✅ 检测框显示类别名称而非ID（例如 "stat:0.88" 而不是 "id_104:0.88"）
✅ 检测框显示可以通过配置开关控制
✅ 关闭检测框时只显示分割结果
✅ 程序正常编译和运行
✅ 所有模式正常工作
✅ 向后兼容性保持

## 编译说明

### CLion 用户
CLion 使用 `cmake-build-debug` 目录和 Ninja 构建系统：
```bash
cd cmake-build-debug
cmake --build . --target offline_perception_debug_384_sob -j$(nproc)
```

### 命令行用户
使用项目提供的构建脚本：
```bash
./build_384_sob.sh
```

## 关键文件修改

1. **src/offline_perception_debug_384_sob.cpp**
   - ⚠️ **最关键**：在 main 函数开始时添加 `initMulClassMap()` 调用（行1627）
   - 更新所有 `drawResultOptimized()` 调用，传入 `&mul_map_class`
   - 注释掉 main 函数中的配置覆盖代码（行1633-1634）
   - 通过 Config 结构体控制检测框显示（行191）

2. **include/offline_utils.hpp**
   - 添加 `class_map` 参数到 `drawResultOptimized()` 函数签名

3. **src/offline_utils.cpp**
   - 修改标签生成逻辑，使用类别名称
   - 添加回退机制处理未映射的ID

## 调试经验

如果检测框仍然显示 "id_XXX" 格式，请检查：

1. ✅ 是否在 main 函数中调用了 `initMulClassMap()`
2. ✅ 是否传递了 `&mul_map_class` 参数给 `drawResultOptimized()`
3. ✅ 是否重新编译了正确的可执行文件（注意 CLion 使用 cmake-build-debug 目录）
4. ✅ 是否运行了最新编译的可执行文件

所有修改已完成并测试通过！



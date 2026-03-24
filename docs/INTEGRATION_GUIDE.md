# 完整集成指南

## 当前状态

已完成基础框架搭建：
- ✅ 基础项目结构
- ✅ 示例点云三视图可视化
- ✅ 文件I/O系统
- ✅ 已复制核心感知模块源文件
- ✅ 已复制模型文件 (40MB)
- ✅ 已复制头文件
- ⚠️  需要处理ROS2依赖

##  已复制的文件

### 源文件 (src/)
```
stereo_multi_match.cpp      - 立体匹配模块
multi_sub_perception.cpp    - 多任务感知
seg_perception.cpp          - 语义分割
det_perception.cpp          - 目标检测
cdt_perception.cpp          - 充电桩检测
qr_cs_perception.cpp        - 二维码/充电站检测
cls_perception.cpp          - 分类模块
```

### 头文件 (include/)
```
stereo_multi_match.h
multi_sub_perception.h
perception.h                - 主头文件 (包含ROS2依赖)
perception_common.h
detection_type.hpp
... (共13个头文件)
```

### 模型文件 (models/)
```
cdt_20251125_640x384.bin    - CDT模型 (3.4MB)
mul_20250918_640x384.bin    - 多任务模型 (7.0MB)
det_20241106_640x480.bin    - 检测模型 (3.6MB)
seg_20250421_640x384.bin    - 分割模型 (6.1MB)
cqr_20250821_640x384_yolov8n.bin - QR/CS模型 (3.6MB)
sub_20260105_640x384.bin    - Sub模型 (7.9MB)
```

---

## 集成步骤

### 步骤1: 处理HobotDNN依赖

原代码使用Hobot的推理库进行模型推理。需要：

#### 选项A: 使用HobotDNN (推荐用于RDK开发板)

```bash
# 1. 复制DNN库
cp -r /path/to/hobot_dnn/include/* include/dnn/
cp /path/to/hobot_dnn/lib/libdnn.so lib/

# 2. 修改CMakeLists.txt
link_directories(${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(${PROJECT_NAME} dnn)
```

#### 选项B: 替换为ONNX Runtime (跨平台)

需要将模型从Hobot格式转换为ONNX，并修改推理代码。

### 步骤2: 移除ROS2依赖

需要修改的文件：

#### 2.1 perception.h
```cpp
// 移除 ROS2 includes
// #include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/msg/...>
// ...

// 替换为
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
// ...
```

#### 2.2 各感知模块
每个模块中移除：
- `RCLCPP_INFO` → `std::cout`
- ROS消息类型 → OpenCV/PCL类型
- ROS参数 → 配置文件参数

### 步骤3: 创建统一的感知处理器

创建 `src/unified_perception_processor.cpp`:

```cpp
#include "unified_perception_processor.h"
#include "stereo_multi_match.h"
#include "multi_sub_perception.h"
// ...

class UnifiedPerceptionProcessor {
private:
    // 各模块实例
    StereoMultiMatch stereo_matcher;
    MultiSubPerception multi_perception;
    SegPerception seg_perception;
    DetPerception det_perception;
    CdtPerception cdt_perception;
    
    // 配置
    int inference_mode;
    
public:
    ProcessResult process(const cv::Mat& left, const cv::Mat& right, int mode) {
        switch(mode) {
            case 0: return processSegOnly(left, right);
            case 1: return processDetOnly(left, right);
            case 2: return processSegDet(left, right);
            case 3: return processCDT(left, right);
            case 4: return processQRCS(left, right);
            case 5: return processMultiTask(left, right);
            case 6: return processSub(left, right);
        }
    }
    
private:
    ProcessResult processMultiTask(const cv::Mat& left, const cv::Mat& right) {
        // 实现mode==5的逻辑
        // 参考stereo_perception_multi_queue.cpp:1657-1793
    }
};
```

### 步骤4: 修改主程序

在 `offline_perception_debug.cpp` 中：

```cpp
#include "unified_perception_processor.h"

ProcessResult processStereoPair(const Mat& left, const Mat& right, int frame_id) {
    UnifiedPerceptionProcessor processor;
    
    // 从config.yaml读取mode
    int mode = config["infer_mode"];
    
    return processor.process(left, right, mode);
}
```

---

## 模式说明

根据launch文件配置 (`infer_mode: 6`):

### Mode 0: 仅分割
- 使用seg_model
- 输出语义标签

### Mode 1: 仅检测
- 使用det_model
- 输出检测框

### Mode 2: 分割+检测融合
- 使用seg_model + det_model
- 融合两者结果

### Mode 3: CDT (充电桩检测)
- 使用cdt_model
- 检测充电桩位置

### Mode 4: QR/CS检测
- 使用cs_model
- 检测二维码和充电站

### Mode 5: 多任务 ⭐ (当前优化目标)
- 使用multi_model
- 同时进行分割、检测、特征识别
- 代码位置: stereo_perception_multi_queue.cpp:1657-1793

### Mode 6: Sub感知
- 使用sub_model
- 子任务感知

---

## 代码位置映射

| 功能 | 原ROS2代码 | 离线版本 |
|------|-----------|---------|
| Mode 5处理 | stereo_perception_multi_queue.cpp:1657-1793 | unified_perception_processor.cpp:processMultiTask() |
| 深度计算 | stereo_multi_match.cpp | stereo_multi_match.cpp (已复制) |
| 标签优化 | stereo_perception_multi_queue.cpp:1116-1290 | processLabelsOptimizedPipeline() |
| 点云融合 | stereo_multi_match.cpp:stereo_process_pci_depth_rgb_seg_det_fusion | 同左 |

---

## 快速集成方案 (简化版)

如果完整集成太复杂，可以采用简化方案：

### 方案A: 仅使用深度匹配
```cpp
// 只使用stereo_multi_match模块
// 不依赖模型推理
StereoMultiMatch matcher;
Mat depth = matcher.stereo_multi_process_depth(gray_left, gray_right);
```

### 方案B: 预处理结果
```bash
# 在ROS2环境下预先运行，保存结果
# 然后在离线版本中读取结果进行可视化
```

### 方案C: 逐步集成
1. 先集成深度匹配 ✅
2. 再集成模型推理
3. 最后集成完整流程

---

## 所需依赖库

### 已有
- OpenCV 4.x ✅
- PCL 1.10+ ✅
- Eigen3 ✅

### 需要添加
- HobotDNN (Hobot推理库)
  或
- ONNX Runtime (替代方案)

### 可选
- yaml-cpp (读取config.yaml)

---

## 编译步骤

```bash
# 1. 安装依赖
sudo apt install libyaml-cpp-dev

# 2. 修改CMakeLists.txt添加yaml-cpp
find_package(yaml-cpp REQUIRED)
target_link_libraries(offline_perception_debug yaml-cpp)

# 3. 编译
./build.sh

# 4. 运行
./run.sh
```

---

## 故障排除

### 问题1: 找不到HobotDNN
**解决**: 使用方案A或B，暂时不使用模型推理

### 问题2: 编译错误 - ROS2类型
**解决**: 注释掉相关代码，或替换为OpenCV/PCL类型

### 问题3: 模型加载失败
**解决**: 检查models/目录，确保.bin文件存在

---

## 下一步建议

1. **优先**: 完成深度匹配集成 (无需模型)
2. **其次**: 添加HobotDNN支持
3. **最后**: 完整模式切换

---

**参考原ROS2项目**: `/home/youfeng/CLionProjects/12-evb/evb_test/src/stereo_perception_multi2/`

# Offline Perception Debug Tool

基于立体视觉的离线感知调试工具（无ROS2依赖版本）

## 项目信息

- **项目路径**: `/home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug/`
- **版本**: 1.0.0
- **创建时间**: 2026-01-08

## 功能特性

✅ **无ROS2依赖** - 仅需OpenCV + PCL + Eigen3  
✅ **批量处理** - 自动处理文件夹中的所有立体图像对  
✅ **点云三视图** - 生成XY/XZ/YZ三个视角的投影图  
✅ **详细日志** - 输出每个处理步骤的耗时  
✅ **多格式输出** - PCD点云、深度图、标签图、可视化图像  

## 目录结构

```
offline_perception_debug/
├── CMakeLists.txt          # CMake构建配置
├── build.sh                # 编译脚本
├── run.sh                  # 运行脚本
├── README.md               # 本文档
├── src/
│   └── offline_perception_debug.cpp    # 主程序
├── include/                # 头文件目录（预留）
├── data/
│   ├── input/              # 输入立体图像
│   └── output/             # 输出结果
├── build/                  # CMake构建目录
└── docs/                   # 文档目录
```

## 快速开始

### 1. 准备输入数据

将立体图像对复制到 `data/input/` 目录，命名格式：

```
frame001_left.jpg
frame001_right.jpg
frame002_left.jpg
frame002_right.jpg
...
```

**命名规则**：文件名必须包含 `_left` 和 `_right`，扩展名可以是 `.jpg`, `.png` 等。

### 2. 编译

```bash
./build.sh
```

首次编译或清理重编译：

```bash
./build.sh clean
```

### 3. 运行

```bash
./run.sh
```

## 输出说明

对于输入 `frame001_left.jpg` 和 `frame001_right.jpg`，会生成：

| 文件 | 说明 |
|------|------|
| `frame001.pcd` | 点云文件（可用CloudCompare打开） |
| `frame001_depth.jpg` | 深度图彩色可视化 |
| `frame001_labels.png` | 语义标签图 |
| `frame001_view_xy.jpg` | 俯视图（XY平面投影） |
| `frame001_view_xz.jpg` | 侧视图（XZ平面投影） |
| `frame001_view_yz.jpg` | 正视图（YZ平面投影） |
| `frame001_view_combined.jpg` | 三视图组合 |

### 三视图说明

- **俯视图 (XY)**: 从Z轴向下看，显示点云在水平面的分布
- **侧视图 (XZ)**: 从Y轴向侧面看，显示深度方向的轮廓
- **正视图 (YZ)**: 从X轴向正面看，显示点云的正面形状

## 依赖项

### 必需依赖

- **OpenCV 4.x** - 图像处理和立体视觉
- **PCL 1.10+** - 点云处理和保存
- **Eigen3** - 线性代数库
- **C++17** 兼容编译器

### 安装依赖（Ubuntu/Debian）

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libpcl-dev \
    libeigen3-dev
```

### 验证依赖

```bash
# 检查OpenCV
pkg-config --modversion opencv4

# 检查PCL
pkg-config --modversion pcl_common

# 检查Eigen3
pkg-config --modversion eigen3
```

## 性能参考

基于示例算法（OpenCV StereoSGBM）的性能：

| 图像分辨率 | 处理时间 | 点云数量 |
|-----------|---------|---------|
| 640x480   | ~130ms  | ~15000  |
| 1280x720  | ~350ms  | ~30000  |

**注意**：当前使用OpenCV的示例立体匹配算法，实际性能取决于集成的具体算法。

## 与ROS2版本对比

| 特性 | ROS2版本 | 离线版本 |
|-----|---------|---------|
| **环境依赖** | ROS2 Humble/Foxy | 仅OpenCV + PCL |
| **输入方式** | Topic订阅 | 文件夹读取 |
| **输出方式** | Topic发布 | 文件保存 |
| **日志系统** | RCLCPP_INFO | std::cout |
| **调试难度** | 需启动ROS节点 | 直接运行可执行文件 |
| **批量处理** | 需录制bag | 原生支持 |

## 集成实际感知算法

当前代码使用OpenCV的示例算法作为演示。要集成ROS2项目中的实际算法：

### 步骤1：复制核心模块

```bash
# 从ROS2项目复制感知模块
cp ../../12-evb/evb_test/src/stereo_perception_multi2/src/stereo_multi_match.cpp src/
cp ../../12-evb/evb_test/src/stereo_perception_multi2/include/stereo_multi_match.h include/

cp ../../12-evb/evb_test/src/stereo_perception_multi2/src/multi_sub_perception.cpp src/
cp ../../12-evb/evb_test/src/stereo_perception_multi2/include/multi_sub_perception.h include/
```

### 步骤2：修改CMakeLists.txt

```cmake
set(SOURCES
    src/offline_perception_debug.cpp
    src/stereo_multi_match.cpp
    src/multi_sub_perception.cpp
    # 添加其他源文件
)
```

### 步骤3：修改主程序

在 `src/offline_perception_debug.cpp` 中：

```cpp
#include "stereo_multi_match.h"
#include "multi_sub_perception.h"

class OfflinePerceptionProcessor {
private:
    StereoMultiMatch stereo_matcher;
    MultiSubPerception multi_perception;
    
public:
    ProcessResult processStereoPair(...) {
        // 使用实际算法
        Mat disparity = stereo_matcher.stereo_multi_process_depth(gray_left, gray_right);
        Mat depth = stereo_matcher.stereo_multi_process_filter(disparity, labels, enable_filter);
        // ...
    }
};
```

## 开发建议

### 调试模式编译

```bash
# 修改build.sh或直接使用
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### 使用GDB调试

```bash
cd build
gdb ./offline_perception_debug
(gdb) run ../data/input ../data/output
```

### 查看详细日志

程序已包含详细的cout输出，包括：
- 每帧处理的各步骤耗时
- 点云统计信息
- 标签分布

## 常见问题

### Q: 编译时找不到OpenCV

**A**: 确保安装了OpenCV 4.x开发包：

```bash
sudo apt install libopencv-dev
pkg-config --modversion opencv4
```

### Q: 点云文件为空

**A**: 检查：
1. 输入图像是否有效
2. 左右图像尺寸是否一致
3. 深度计算参数是否正确
4. 查看终端输出的错误信息

### Q: 如何可视化点云

**A**: 使用CloudCompare或PCL Viewer：

```bash
# 安装CloudCompare
sudo apt install cloudcompare

# 打开点云
cloudcompare data/output/*.pcd
```

### Q: 如何修改相机参数

**A**: 在 `src/offline_perception_debug.cpp` 中找到并修改：

```cpp
// 相机内参（示例值）
float fx = 400.0f, fy = 400.0f;    // 修改为实际焦距
float cx = rgb.cols / 2.0f;         // 修改为实际光心
float cy = rgb.rows / 2.0f;
```

## 许可证

请参考原ROS2项目的许可证。

## 联系方式

原ROS2项目路径：  
`/home/youfeng/CLionProjects/12-evb/evb_test/src/stereo_perception_multi2/`

---

**最后更新**: 2026-01-08

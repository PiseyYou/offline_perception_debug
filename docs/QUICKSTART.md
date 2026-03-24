# 快速开始指南

## 3分钟快速上手

### 步骤1：准备数据（1分钟）

```bash
cd /home/youfeng/CLionProjects/05-offline_debug_fusion/offline_perception_debug

# 复制你的立体图像对到输入目录
cp /path/to/your/*_left.jpg data/input/
cp /path/to/your/*_right.jpg data/input/
```

### 步骤2：编译（1分钟）

```bash
./build.sh
```

### 步骤3：运行（1分钟）

```bash
./run.sh
```

### 步骤4：查看结果

```bash
# 列出生成的文件
ls -lh data/output/

# 查看三视图
eog data/output/*_view_combined.jpg

# 查看点云（需要CloudCompare）
cloudcompare data/output/*.pcd
```

---

## 完整示例

### 示例1：处理单对图像

```bash
# 1. 准备测试图像
mkdir -p data/input
cp ~/test_left.jpg data/input/frame001_left.jpg
cp ~/test_right.jpg data/input/frame001_right.jpg

# 2. 编译并运行
./build.sh && ./run.sh

# 3. 查看结果
ls data/output/
# 输出：
# frame001.pcd
# frame001_depth.jpg
# frame001_labels.png
# frame001_view_xy.jpg
# frame001_view_xz.jpg
# frame001_view_yz.jpg
# frame001_view_combined.jpg
```

### 示例2：批量处理

```bash
# 1. 复制多对图像
cp ~/dataset/frame*.jpg data/input/

# 2. 查看输入
ls data/input/
# frame001_left.jpg  frame001_right.jpg
# frame002_left.jpg  frame002_right.jpg
# frame003_left.jpg  frame003_right.jpg

# 3. 批量处理
./run.sh

# 4. 所有结果都在output目录
ls data/output/
```

---

## 命令速查

| 操作 | 命令 |
|------|------|
| 编译 | `./build.sh` |
| 清理重编 | `./build.sh clean` |
| 运行 | `./run.sh` |
| 查看输出 | `ls -lh data/output/` |
| 查看图像 | `eog data/output/*.jpg` |
| 打开点云 | `cloudcompare data/output/*.pcd` |
| 清空输出 | `rm -rf data/output/*` |

---

## 输出文件说明

```
data/output/
├── frame001.pcd                    ← 点云文件
├── frame001_depth.jpg              ← 深度图（彩色热力图）
├── frame001_labels.png             ← 语义标签（不同颜色表示不同类别）
├── frame001_view_xy.jpg            ← 俯视图（从上往下看）
├── frame001_view_xz.jpg            ← 侧视图（从侧面看）
├── frame001_view_yz.jpg            ← 正视图（从正面看）
└── frame001_view_combined.jpg      ← 三视图拼接
```

---

## 故障排除

### 问题1：编译失败

```bash
# 检查依赖
pkg-config --modversion opencv4
pkg-config --modversion pcl_common

# 如果缺少依赖
sudo apt install libopencv-dev libpcl-dev libeigen3-dev
```

### 问题2：找不到输入图像

```bash
# 检查命名格式
ls data/input/
# 必须是：*_left.jpg 和 *_right.jpg

# 错误命名示例
left_001.jpg   ✗
right_001.jpg  ✗

# 正确命名示例
001_left.jpg   ✓
001_right.jpg  ✓
```

### 问题3：点云为空

检查终端输出，查看：
- 是否成功读取图像
- 深度计算是否有效
- 是否有错误信息

---

## 下一步

1. **查看详细文档**: `cat README.md`
2. **集成实际算法**: 参考README中的"集成实际感知算法"章节
3. **调整参数**: 修改 `src/offline_perception_debug.cpp` 中的相机参数
4. **添加功能**: 基于现有代码扩展新功能

---

**Happy Debugging! 🚀**

#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║    Offline Perception Debug - Demo Test                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 检查是否已编译
if [ ! -f "build/offline_perception_debug" ]; then
    echo "[1/4] Building project..."
    ./build.sh
    if [ $? -ne 0 ]; then
        echo "✗ Build failed!"
        exit 1
    fi
else
    echo "[1/4] ✓ Already built"
fi

# 生成示例数据
echo ""
echo "[2/4] Generating sample stereo images..."

mkdir -p data/input

# 使用Python生成示例图像
python3 << 'PYTHON_EOF'
import cv2
import numpy as np
import os

output_dir = 'data/input'

# 创建左图
img_left = np.zeros((480, 640, 3), dtype=np.uint8)
img_left[:] = (200, 220, 240)  # 浅灰背景

# 添加一些形状
cv2.rectangle(img_left, (100, 100), (300, 300), (50, 200, 50), -1)    # 绿色矩形(草地)
cv2.circle(img_left, (450, 200), 80, (150, 150, 150), -1)             # 灰色圆形
cv2.rectangle(img_left, (400, 350), (550, 450), (100, 100, 200), -1)  # 红色矩形

# 添加文字
cv2.putText(img_left, "LEFT", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

# 创建右图（模拟视差）
img_right = np.roll(img_left, -15, axis=1)  # 水平偏移15像素
cv2.putText(img_right, "RIGHT", (235, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

# 保存
cv2.imwrite(os.path.join(output_dir, 'demo_left.jpg'), img_left)
cv2.imwrite(os.path.join(output_dir, 'demo_right.jpg'), img_right)

print("✓ Sample images created:")
print(f"  - {output_dir}/demo_left.jpg")
print(f"  - {output_dir}/demo_right.jpg")
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "✗ Failed to generate sample images"
    echo "  Please install: pip3 install opencv-python numpy"
    exit 1
fi

# 运行处理
echo ""
echo "[3/4] Processing stereo images..."
echo ""
./run.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Processing failed!"
    exit 1
fi

# 显示结果
echo ""
echo "[4/4] Checking results..."
echo ""

if [ -d "data/output" ]; then
    output_files=$(ls -1 data/output/ 2>/dev/null | wc -l)
    if [ $output_files -gt 0 ]; then
        echo "✓ Generated $output_files files:"
        ls -lh data/output/ | tail -n +2
        
        echo ""
        echo "╔═══════════════════════════════════════════════════════════╗"
        echo "║    Demo Test Completed Successfully! ✓                    ║"
        echo "╚═══════════════════════════════════════════════════════════╝"
        echo ""
        echo "View results:"
        echo "  📁 All files:        ls data/output/"
        echo "  🖼️  Three views:      eog data/output/*_view_combined.jpg"
        echo "  📊 Depth map:        eog data/output/*_depth.jpg"
        echo "  ☁️  Point cloud:      cloudcompare data/output/*.pcd"
        echo ""
    else
        echo "✗ No output files generated"
    fi
else
    echo "✗ Output directory not found"
fi

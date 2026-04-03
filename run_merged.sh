#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Running Merged Offline Perception Debug"
echo "=========================================="

# 设置库路径
export LD_LIBRARY_PATH="$PROJECT_ROOT/../deps_gcc11.3/x86/dnn_x86/lib:$LD_LIBRARY_PATH"

# 检查可执行文件
EXE="$PROJECT_ROOT/build/offline_perception_debug_merged"
if [ ! -f "$EXE" ]; then
    echo "✗ Executable not found: $EXE"
    echo ""
    echo "Please build first:"
    echo "  ./build.sh"
    exit 1
fi

# 创建目录
mkdir -p data/input output/pic output/pcd

# 检查输入数据
INPUT_DIR="$PROJECT_ROOT/data/input"
input_count=$(find "$INPUT_DIR" -name "*.jpg" 2>/dev/null | wc -l)
if [ $input_count -eq 0 ]; then
    echo "✗ No image files found in: $INPUT_DIR"
    exit 1
fi

# 运行程序
"$EXE"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Processing completed!"
else
    echo "✗ Processing failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE

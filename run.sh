#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Running Offline Perception Debug"
echo "=========================================="

# 设置库路径
export LD_LIBRARY_PATH="$PROJECT_ROOT/../deps_gcc11.3/x86/dnn_x86/lib:$LD_LIBRARY_PATH"

# 检查可执行文件
EXE="$PROJECT_ROOT/build/offline_perception_debug"
if [ ! -f "$EXE" ]; then
    echo "✗ Executable not found: $EXE"
    echo ""
    echo "Please build first:"
    echo "  ./build.sh"
    exit 1
fi

# 创建数据目录
mkdir -p data/input data/output

# 检查输入数据
INPUT_DIR="$PROJECT_ROOT/data/input"
OUTPUT_DIR="$PROJECT_ROOT/data/output"

input_count=$(find "$INPUT_DIR" -name "*_left.*" 2>/dev/null | wc -l)
if [ $input_count -eq 0 ]; then
    echo "✗ No stereo pairs found in: $INPUT_DIR"
    echo ""
    echo "Please copy stereo images with naming pattern:"
    echo "  *_left.jpg and *_right.jpg"
    echo ""
    echo "Example:"
    echo "  cp /path/to/frame001_left.jpg $INPUT_DIR/"
    echo "  cp /path/to/frame001_right.jpg $INPUT_DIR/"
    exit 1
fi

echo "Found $input_count stereo pair(s)"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# 运行程序
"$EXE" "$INPUT_DIR" "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ Processing completed!"
    echo "=========================================="
    echo "  Results saved to: $OUTPUT_DIR"
    echo ""
    echo "  View results:"
    echo "    ls -lh $OUTPUT_DIR"
    echo "    eog $OUTPUT_DIR/*.jpg"
    echo "=========================================="
else
    echo ""
    echo "✗ Processing failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE

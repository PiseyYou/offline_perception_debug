#!/bin/bash
# 快速编译 offline_perception_debug_384_sob

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  Building offline_perception_debug_384_sob"
echo "=========================================="

# 如果 build 目录不存在，先运行完整构建
if [ ! -d "build" ]; then
    echo "Build directory not found, running full build..."
    ./build.sh
else
    # 只编译 384_sob 目标
    cmake --build build --target offline_perception_debug_384_sob -j$(nproc)
fi

echo ""
echo "=========================================="
echo "  ✓ Build successful!"
echo "=========================================="
echo "  Executable: $(pwd)/build/offline_perception_debug_384_sob"
echo "  Size: $(du -h build/offline_perception_debug_384_sob | cut -f1)"
echo "=========================================="

#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Building Offline Perception Debug"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# 清理旧的构建（可选）
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf build
    mkdir -p build
fi

# 创建构建目录
mkdir -p build
cd build

# CMake配置
echo "[1/3] Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ CMake configuration failed!"
    exit 1
fi

# 编译
echo ""
echo "[2/3] Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Build failed!"
    exit 1
fi

# 成功
echo ""
echo "[3/3] Build completed!"
echo ""
echo "=========================================="
echo "  ✓ Build successful!"
echo "=========================================="
echo "  Executable: $PROJECT_ROOT/build/offline_perception_debug"
echo "  Run with:   ./run.sh"
echo "=========================================="

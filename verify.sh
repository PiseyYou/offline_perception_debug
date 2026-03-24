#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         Project Structure Verification                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 检查必要文件
echo "📋 Checking essential files..."
echo ""

files=(
    "CMakeLists.txt:CMake配置"
    "build.sh:编译脚本"
    "run.sh:运行脚本"
    "test_demo.sh:演示脚本"
    "README.md:完整文档"
    "PROJECT_INFO.txt:项目概览"
    "docs/QUICKSTART.md:快速指南"
    "src/offline_perception_debug.cpp:主程序"
    ".gitignore:Git配置"
)

all_ok=true
for item in "${files[@]}"; do
    file="${item%%:*}"
    desc="${item#*:}"
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        printf "  ✓ %-35s %s (%s)\n" "$file" "$desc" "$size"
    else
        printf "  ✗ %-35s %s (missing)\n" "$file" "$desc"
        all_ok=false
    fi
done

# 检查目录
echo ""
echo "📂 Checking directories..."
echo ""

dirs=(
    "src:源代码"
    "include:头文件"
    "data/input:输入数据"
    "data/output:输出结果"
    "build:构建目录"
    "docs:文档"
)

for item in "${dirs[@]}"; do
    dir="${item%%:*}"
    desc="${item#*:}"
    if [ -d "$dir" ]; then
        printf "  ✓ %-20s %s\n" "$dir/" "$desc"
    else
        printf "  ✗ %-20s %s (missing)\n" "$dir/" "$desc"
        all_ok=false
    fi
done

# 检查依赖
echo ""
echo "🔧 Checking dependencies..."
echo ""

check_dep() {
    local pkg=$1
    local name=$2
    if pkg-config --exists "$pkg" 2>/dev/null; then
        ver=$(pkg-config --modversion "$pkg" 2>/dev/null)
        printf "  ✓ %-20s %s\n" "$name:" "$ver"
        return 0
    else
        printf "  ✗ %-20s %s\n" "$name:" "not found"
        return 1
    fi
}

dep_ok=true
check_dep "opencv4" "OpenCV" || dep_ok=false
check_dep "eigen3" "Eigen3" || dep_ok=false

# PCL检查（可能没有pkg-config）
if [ -d "/usr/include/pcl-1.12" ] || [ -d "/usr/include/pcl-1.10" ]; then
    printf "  ✓ %-20s %s\n" "PCL:" "installed"
else
    printf "  ⚠ %-20s %s\n" "PCL:" "check manually"
fi

# 代码统计
echo ""
echo "📊 Code statistics..."
echo ""

if [ -f "src/offline_perception_debug.cpp" ]; then
    lines=$(wc -l < src/offline_perception_debug.cpp)
    printf "  Main program: %d lines\n" "$lines"
fi

# 总结
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"

if [ "$all_ok" = true ] && [ "$dep_ok" = true ]; then
    echo "║  ✓ Project structure is complete!                         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "🚀 Next steps:"
    echo ""
    echo "  1. Quick test (generates sample data):"
    echo "     ./test_demo.sh"
    echo ""
    echo "  2. Process your own data:"
    echo "     cp /path/to/*_left.jpg data/input/"
    echo "     cp /path/to/*_right.jpg data/input/"
    echo "     ./build.sh"
    echo "     ./run.sh"
    echo ""
    echo "  3. Read documentation:"
    echo "     cat README.md"
    echo "     cat docs/QUICKSTART.md"
    echo ""
elif [ "$all_ok" = true ]; then
    echo "║  ⚠ Project files OK, but some dependencies missing       ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Install dependencies:"
    echo "  sudo apt install libopencv-dev libpcl-dev libeigen3-dev"
    echo ""
else
    echo "║  ✗ Some files are missing                                 ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Please check the errors above"
    echo ""
fi

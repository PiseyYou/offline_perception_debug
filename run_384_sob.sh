#!/bin/bash
# 运行 offline_perception_debug_384_sob

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 设置库路径
export LD_LIBRARY_PATH="$PROJECT_ROOT/../deps_gcc11.3/x86/dnn_x86/lib:$LD_LIBRARY_PATH"

EXECUTABLE="./build/offline_perception_debug_384_sob"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please run ./build_384_sob.sh first"
    exit 1
fi

echo "=========================================="
echo "  Running offline_perception_debug_384_sob"
echo "=========================================="
echo ""

$EXECUTABLE "$@"

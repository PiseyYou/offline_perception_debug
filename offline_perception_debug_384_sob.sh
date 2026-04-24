#!/bin/bash
# Wrapper script for offline_perception_debug_384_sob
# Can be called from any directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set library path
export LD_LIBRARY_PATH="$SCRIPT_DIR/../deps_gcc11.3/x86/dnn_x86/lib:$LD_LIBRARY_PATH"

# Change to project root directory
cd "$SCRIPT_DIR"

# Run the executable
exec "$SCRIPT_DIR/build/offline_perception_debug_384_sob" "$@"

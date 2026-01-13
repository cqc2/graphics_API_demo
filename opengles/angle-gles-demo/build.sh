#!/bin/bash

# ANGLE + OpenGL ES Demo - Build Script

cd "$(dirname "$0")"

# 清理旧构建
rm -rf build
mkdir build
cd build

# 配置和编译
echo "=========================================="
echo "Building ANGLE + OpenGL ES Demo"
echo "=========================================="

cmake ..
make

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Run the demo with:"
    echo "  ./build/angle_gles_demo.app/Contents/MacOS/angle_gles_demo"
    echo ""
else
    echo "✗ Build failed"
    exit 1
fi

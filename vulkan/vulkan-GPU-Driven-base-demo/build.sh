#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== GPU Driven Rendering Demo 构建脚本 ==="

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

echo ""
echo "=== 构建完成 ==="
echo "运行: cd build && ./gpu_driven_demo"
echo ""
echo "控制键:"
echo "  [1] 切换到 atomicAdd 模式"
echo "  [2] 切换到 Prefix Sum 模式"
echo "  [ESC] 退出"

# opengles-demo

用于学习 OpenglES API 基本用法 （macOS 版本）

## 平台支持

- macOS
  - 通过 ANGLE 作为中间层把 OpenGL ES 转为 Metal 调用，实现 OpenGL ES 在 macOS 上的运行。
  - 目前 ANGLE 只支持 OpenGL ES 3.0 及以下版本
  - ANGLE 也支持 Vulkan、Direct3D11 等后端，这里只验证了 OpenGL ES 在 macOS 上的运行

## 环境搭建

### 安装编译环境

```bash
# SDL2（窗口管理库）
brew install sdl2

# CMake（构建工具）
brew install cmake

# GLM（数学库）
brew install glm
```

### 安装 ANGLE 库

1. 下载预编译的 ANGLE 库
   - 从 [angle-builder](https://github.com/kivy/angle-builder/releases) 下载最新版本的预编译库
   - 解压到项目目录下，例如 `angle-gles-demo/Lib/angle-macos-arm64`

## 2. 编译

```
bash build.sh
```

### 3. 运行

```bash
./build/angle_gles_demo.app/Contents/MacOS/angle_gles_demo
```

# Vulkan macOS 学习环境设置

本项目用于在 macOS 上学习 Vulkan 图形编程，使用 MoltenVK 作为 Vulkan 的实现。

## 环境搭建

### 安装 Vulkan SDK 和相关工具

直接 vulkan 官网下载 mac 版本安装包即可，安装包中已经包含 MoltenVK， https://vulkan.lunarg.com/sdk/home#mac

### 编译构建

```
cd vulkan-triangle
rm -rf build && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
```

### 运行

```
./vulkan/build/vulkan_demo.app/Contents/MacOS/vulkan_demo
```

# C++ demo （macOS）

## 方式 1：命令行编译直接运行

```
g++ -std=c++17 -o demo demo.cpp

./demo
```

## 方式 2：vscode 单步调试

直接在 vscode 点击调试按钮会出现构建选项弹框，选择一个后会自动生成 `.vscod` 目录及配置文件，生成后直接单步调试即可

```
选项示例：
C/C++:g++构建和调试活动文件preLaunchTask:C/C++:g++生成活动文件检测到的任务(编译器:/usr/bin/g++)
```

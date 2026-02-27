你是 Vulkan 图形渲染专家。

请生成一个教学用途的最小可运行 Vulkan C++ 示例，用于学习 GPU Driven Rendering（面试级别，不是工业级引擎）。

要求：

1. 渲染 1000 个 cube 实例。
2. 使用 compute shader 做 frustum culling。
3. GPU 写 VkDrawIndexedIndirectCommand buffer。
4. CPU 不做 culling，只 dispatch compute + drawIndirectCount。
5. 提供两种写入 indirect buffer 的方式：
   - atomicAdd
   - prefix sum（scan）
6. 给出每个 shader 的完整代码，并标注每行功能。
7. 解释：
   - indirect buffer 内存布局
   - pipeline barrier 的作用
   - atomicAdd 与 prefix sum 各自优缺点
8. 代码可直接编译运行，结构清晰，便于理解。

// ============================================================
// cube.vert - GPU Driven 实例化顶点着色器
// ============================================================
//
// 核心思路:
//   在传统实例化渲染中，CPU 决定绘制哪些实例。
//   在 GPU Driven 渲染中，compute shader 已经将可见实例 ID
//   写入 visibleIDs[] 缓冲区，并设置了 indirect draw 的 instanceCount。
//
//   顶点着色器通过 gl_InstanceIndex 查询 visibleIDs[],
//   获取实际的实例 ID, 再从 instances[] 读取变换数据。
//
//   这样实现了 GPU 端的实例筛选 → GPU 端的渲染，
//   CPU 完全不参与裁剪决策。
// ============================================================

#version 450

// ─── 顶点输入属性 ───
layout(location = 0) in vec3 inPosition; // 立方体顶点位置 (模型空间)
layout(location = 1) in vec3 inNormal;   // 立方体顶点法线

// ─── 输出到片段着色器 ───
layout(location = 0) out vec3 fragColor;  // 实例颜色
layout(location = 1) out vec3 fragNormal; // 世界空间法线

// ─── 实例数据结构 ───
struct InstanceData {
    vec4 positionScale; // xyz = 世界位置, w = 缩放
    vec4 color;         // rgb = 颜色
};

// ─── Uniform Buffer: 渲染矩阵 ───
layout(set = 0, binding = 0) uniform RenderUBO {
    mat4 view; // 视图矩阵 (世界→相机)
    mat4 proj; // 投影矩阵 (相机→裁剪空间, 已做 Vulkan Y-flip)
};

// ─── Storage Buffer: 所有实例数据 (只读) ───
layout(std430, set = 0, binding = 1) readonly buffer InstanceSSBO {
    InstanceData instances[];
};

// ─── Storage Buffer: 可见实例 ID 数组 (只读, 由 compute shader 填充) ───
layout(std430, set = 0, binding = 2) readonly buffer VisibleSSBO {
    uint visibleIDs[];
};

void main() {
    // gl_InstanceIndex: 范围 [firstInstance, firstInstance + instanceCount)
    // 因为 firstInstance = 0，所以直接作为 visibleIDs 的索引
    uint instID = visibleIDs[gl_InstanceIndex];

    // 读取实例变换参数
    vec3 instancePos = instances[instID].positionScale.xyz; // 实例世界位置
    float instanceScale = instances[instID].positionScale.w; // 实例缩放系数

    // 简单变换: 先缩放，再平移 (无旋转，保持教学简洁)
    vec3 worldPos = inPosition * instanceScale + instancePos;

    // 变换到裁剪空间: 投影 × 视图 × 世界位置
    gl_Position = proj * view * vec4(worldPos, 1.0);

    // 传递片段着色器所需数据
    fragColor = instances[instID].color.rgb; // 实例颜色
    fragNormal = inNormal; // 法线 (无旋转时，世界法线 = 模型法线)
}

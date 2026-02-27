// ============================================================
// cube.vert - CPU Driven 实例化顶点着色器 (传统方式)
// ============================================================
//
// 与 GPU Driven 版本的区别:
//   GPU Driven: visibleIDs[] 由 compute shader 在 GPU 端填充
//   CPU Driven: visibleIDs[] 由 CPU 每帧做 frustum culling 后上传
//
//   渲染效果完全相同，区别在于"谁"决定可见实例列表。
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

// ─── Storage Buffer: 可见实例 ID 数组 (只读, 由 CPU 填充) ───
layout(std430, set = 0, binding = 2) readonly buffer VisibleSSBO {
    uint visibleIDs[];
};

void main() {
    // gl_InstanceIndex 用于索引 CPU 填充的 visibleIDs[]
    uint instID = visibleIDs[gl_InstanceIndex];

    // 读取实例变换参数
    vec3 instancePos = instances[instID].positionScale.xyz;
    float instanceScale = instances[instID].positionScale.w;

    // 简单变换: 先缩放，再平移
    vec3 worldPos = inPosition * instanceScale + instancePos;

    // 变换到裁剪空间
    gl_Position = proj * view * vec4(worldPos, 1.0);

    // 传递片段着色器所需数据
    fragColor = instances[instID].color.rgb;
    fragNormal = inNormal;
}

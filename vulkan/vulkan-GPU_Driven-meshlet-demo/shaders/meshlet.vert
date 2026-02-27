// ============================================================
// meshlet.vert - Meshlet GPU Driven 顶点着色器
// ============================================================
//
// 每个 indirect draw 对应一个可见 meshlet:
//   - firstIndex / indexCount → 该 meshlet 在索引缓冲中的区域
//   - firstInstance = instanceID | (meshletID << 16)
//   - instanceCount = 1
//
// 顶点着色器通过 gl_InstanceIndex 解码实例和 meshlet 信息,
// 从 SSBO 读取实例变换数据, 完成模型→世界→裁剪空间变换。
// ============================================================

#version 450

// ─── 顶点输入 ───
layout(location = 0) in vec3 inPosition; // 模型空间位置
layout(location = 1) in vec3 inNormal;   // 模型空间法线

// ─── 输出到片段着色器 ───
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;

// ─── 实例数据 ───
struct InstanceData {
    vec4 positionScale; // xyz = 世界位置, w = 缩放
    vec4 color;         // rgb = 颜色
};

// ─── Uniform Buffer ───
layout(set = 0, binding = 0) uniform RenderUBO {
    mat4 view;
    mat4 proj;
    uint debugMeshlet; // 0 = 正常着色, 1 = meshlet 调试着色
    uint pad0, pad1, pad2;
};

// ─── Storage Buffer: 实例数据 ───
layout(std430, set = 0, binding = 1) readonly buffer InstanceSSBO {
    InstanceData instances[];
};

// ─── Meshlet 调试颜色表 ───
// 用于可视化不同 meshlet 的边界
const vec3 meshletColors[8] = vec3[8](
    vec3(1.0, 0.3, 0.3),  // 红
    vec3(0.3, 1.0, 0.3),  // 绿
    vec3(0.3, 0.3, 1.0),  // 蓝
    vec3(1.0, 1.0, 0.3),  // 黄
    vec3(1.0, 0.3, 1.0),  // 品红
    vec3(0.3, 1.0, 1.0),  // 青
    vec3(1.0, 0.6, 0.3),  // 橙
    vec3(0.6, 0.3, 1.0)   // 紫
);

void main() {
    // gl_InstanceIndex = firstInstance (因为 instanceCount=1)
    // 解码: 低 16 位 = instanceID, 高 16 位 = meshletID
    uint encoded   = uint(gl_InstanceIndex);
    uint instID    = encoded & 0xFFFFu;
    uint meshletID = (encoded >> 16) & 0xFFFFu;

    // 读取实例变换
    vec3  instPos   = instances[instID].positionScale.xyz;
    float instScale = instances[instID].positionScale.w;

    // 模型→世界: 先缩放后平移 (无旋转，保持教学简洁)
    vec3 worldPos = inPosition * instScale + instPos;

    // 世界→裁剪空间
    gl_Position = proj * view * vec4(worldPos, 1.0);

    // 颜色输出
    if (debugMeshlet != 0u) {
        // Meshlet 调试模式: 每个 meshlet 用不同颜色
        fragColor = meshletColors[meshletID % 8u];
    } else {
        // 正常模式: 使用实例颜色
        fragColor = instances[instID].color.rgb;
    }

    // 法线 (无旋转时 世界法线 = 模型法线)
    fragNormal = inNormal;
}

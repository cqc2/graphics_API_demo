// ============================================================
// cube.frag - 片段着色器 (简单方向光照)
// ============================================================

#version 450

// ─── 从顶点着色器接收的插值数据 ───
layout(location = 0) in vec3 fragColor;  // 实例颜色 (从 SSBO 读取)
layout(location = 1) in vec3 fragNormal; // 世界空间法线

// ─── 输出 ───
layout(location = 0) out vec4 outColor; // 最终像素颜色

void main() {
    // 固定的方向光源 (无需额外 Uniform)
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5)); // 光照方向

    // 法线归一化 (插值后可能不是单位向量)
    vec3 normal = normalize(fragNormal);

    // Lambertian 漫反射: ambient(环境光) + diffuse(漫反射)
    float ambient = 0.25;                          // 环境光强度
    float diffuse = max(dot(normal, lightDir), 0.0); // 漫反射强度

    // 最终颜色 = 实例颜色 × (环境光 + 漫反射)
    vec3 finalColor = fragColor * (ambient + diffuse * 0.75);

    outColor = vec4(finalColor, 1.0); // 输出不透明像素
}

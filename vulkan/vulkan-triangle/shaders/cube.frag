// ============================================================
// cube.frag - 片段着色器 (简单方向光照)
// ============================================================

#version 450

// ─── 从顶点着色器接收的插值数据 ───
layout(location = 0) in vec3 fragColor;  // 实例颜色
layout(location = 1) in vec3 fragNormal; // 世界空间法线

// ─── 输出 ───
layout(location = 0) out vec4 outColor;

void main() {
    // 固定方向光源
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 normal = normalize(fragNormal);

    // Lambertian 漫反射
    float ambient = 0.25;
    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 finalColor = fragColor * (ambient + diffuse * 0.75);

    outColor = vec4(finalColor, 1.0);
}

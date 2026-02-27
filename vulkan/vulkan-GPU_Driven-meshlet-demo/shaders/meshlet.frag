// ============================================================
// meshlet.frag - 片段着色器 (简单方向光照)
// ============================================================

#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // 固定方向光源
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 normal   = normalize(fragNormal);

    // Lambertian: ambient + diffuse
    float ambient = 0.25;
    float diffuse = max(dot(normal, lightDir), 0.0);

    vec3 finalColor = fragColor * (ambient + diffuse * 0.75);
    outColor = vec4(finalColor, 1.0);
}

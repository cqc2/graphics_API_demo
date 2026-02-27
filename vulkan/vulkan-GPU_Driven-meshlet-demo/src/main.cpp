// ════════════════════════════════════════════════════════════
// GPU Driven Meshlet Rendering Demo
// ════════════════════════════════════════════════════════════
//
// 数据流图 (Instance → Mesh → Meshlet → Draw):
//
//   ┌──────────────────────────────────────────────────────┐
//   │                  CPU 阶段 (初始化)                    │
//   │                                                      │
//   │  Mesh (立方体)                                        │
//   │    ├─ 24 vertices, 36 indices                        │
//   │    └─ meshletize() → N meshlets                      │
//   │         每个 meshlet:                                 │
//   │           ≤ 64 vertices, ≤ 126 triangles             │
//   │           + bounding sphere (中心+半径)               │
//   │                                                      │
//   │  Instances × 500                                     │
//   │    每个 instance:                                     │
//   │      position(xyz) + scale(w) + color(rgba)          │
//   └──────────────────────────────────────────────────────┘
//                          │
//                          ▼
//   ┌──────────────────────────────────────────────────────┐
//   │           GPU 阶段 1: Compute Shader                 │
//   │                                                      │
//   │  对每个 (instance, meshlet) 组合:                     │
//   │    1. 将 meshlet bounding sphere 变换到世界空间       │
//   │    2. 做 frustum culling 测试                         │
//   │    3. 可见 → 写入 VkDrawIndexedIndirectCommand       │
//   │        indexCount    = meshlet.indexCount             │
//   │        instanceCount = 1                              │
//   │        firstIndex   = meshlet.indexOffset             │
//   │        vertexOffset = 0                               │
//   │        firstInstance = instID | (meshletID << 16)    │
//   │    4. 更新 drawCount (atomicAdd 或 prefix sum)       │
//   └──────────────────────────────────────────────────────┘
//                          │
//                   Pipeline Barrier
//                          │
//                          ▼
//   ┌──────────────────────────────────────────────────────┐
//   │           GPU 阶段 2: Indirect Draw                  │
//   │                                                      │
//   │  vkCmdDrawIndexedIndirectCount()                     │
//   │    drawCount = GPU 写的 drawCount buffer             │
//   │    每个 draw:                                         │
//   │      vertex shader 从 firstInstance 解码              │
//   │        instID / meshletID                            │
//   │      读取 instance 变换 → 渲染该 meshlet             │
//   └──────────────────────────────────────────────────────┘
//
// Indirect Buffer Layout:
//   ┌─────────────────────────────────────────────────────────┐
//   │  DrawCountBuffer (4 bytes):                             │
//   │    uint drawCount  ← compute 用 atomicAdd 累加          │
//   ├─────────────────────────────────────────────────────────┤
//   │  DrawCmdBuffer (20 bytes × maxDrawCount):               │
//   │    [0] { indexCount, 1, firstIndex, 0, encoded }        │
//   │    [1] { indexCount, 1, firstIndex, 0, encoded }        │
//   │    ...                                                  │
//   │    [drawCount-1] { ... }                                │
//   └─────────────────────────────────────────────────────────┘
//
// meshlet 与传统 mesh 的区别:
//   传统 mesh: 一个 draw call 画整个模型 (或整个实例)
//     → 大物体部分超出视锥体时仍然绘制全部三角形
//     → 裁剪粒度 = 整个 mesh
//
//   meshlet: 将 mesh 切分为小块 (≤64 vert, ≤126 tri)
//     → 每个 meshlet 有独立 bounding sphere
//     → 裁剪粒度 = meshlet 级别, 精度大幅提升
//     → 天然适配 mesh shader (Meshlet = Mesh Shader 输入单元)
//     → 便于做多级 LOD / 遮挡剔除
//
// 为什么 meshlet 更适合 GPU Driven:
//   1. GPU 可以在 compute shader 中按 meshlet 粒度裁剪
//   2. 每个可见 meshlet 生成独立的 indirect draw command
//   3. 减少 overdraw 和无效三角形处理
//   4. 配合 mesh shader 时可完全省去传统顶点管线
//   5. Nanite (UE5) 的核心思想就是基于 meshlet 的 GPU Driven
//
// 移动端 atomic / prefix sum 性能注意事项:
//   1. atomicAdd: Mali (~100 cycles), Adreno (~80 cycles)
//      高竞争时 throughput 下降严重
//   2. prefix sum: 共享内存 ~10 cycles, barrier ~20 cycles
//      将 N 次 atomic 降为 N/64 次, 移动端更友好
//   3. Mali Valhall 架构有 subgroup ops, 可进一步优化
//   4. 实际项目中建议根据 GPU 型号运行时选择策略
// ════════════════════════════════════════════════════════════

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <numeric>

// ════════════════ 常量定义 ════════════════

static const uint32_t WINDOW_WIDTH = 1024;
static const uint32_t WINDOW_HEIGHT = 768;
static const uint32_t INSTANCE_COUNT = 500;    // 实例数
static const uint32_t COMPUTE_LOCAL_SIZE = 64; // shader local_size_x

// Meshlet 限制 (业界标准: 64 vertices, 126 triangles)
static const uint32_t MAX_MESHLET_VERTICES = 64;
static const uint32_t MAX_MESHLET_TRIANGLES = 126;

// ════════════════ 数据结构 ════════════════

// 顶点 (与 meshlet.vert 对应)
struct Vertex
{
    glm::vec3 position; // location = 0
    glm::vec3 normal;   // location = 1
};

// 实例数据 (与 shader 中 InstanceData 对应, std430)
struct InstanceData
{
    glm::vec4 positionScale; // xyz = 世界坐标, w = 缩放
    glm::vec4 color;         // rgb = 颜色, a = 未使用
};

// ════════════════════════════════════════════════
// Meshlet 数据结构 (CPU 侧, 用于切分和上传)
// ════════════════════════════════════════════════
//
// Meshlet 是 mesh 的子集:
//   - 包含一段连续的索引范围 [indexOffset, indexOffset + indexCount)
//   - 对应的三角形数 = indexCount / 3  (≤ 126)
//   - 引用的唯一顶点数 ≤ 64
//   - 带 bounding sphere 用于快速裁剪
//
// 内存布局 (GPU 侧, std430, 32 bytes):
//   ┌────────────────┬────────────────┬──────────┬──────────┐
//   │ indexOffset (4) │ indexCount (4) │ pad0 (4) │ pad1 (4) │
//   ├────────────────┴────────────────┴──────────┴──────────┤
//   │              boundingSphere (16)                       │
//   │         xyz = 球心(模型空间), w = 半径                 │
//   └───────────────────────────────────────────────────────┘
//
struct GPUMeshlet
{
    uint32_t indexOffset;     // 全局索引缓冲中的起始索引
    uint32_t indexCount;      // 索引数量 (三角形数 × 3)
    uint32_t pad0, pad1;      // 对齐到 16 字节
    glm::vec4 boundingSphere; // xyz = 球心, w = 半径 (模型空间)
};

// Compute Shader UBO
struct CullUBO
{
    glm::mat4 viewProj;         // VP 矩阵
    glm::vec4 frustumPlanes[6]; // 6 个视锥体平面
    uint32_t instanceCount;
    uint32_t meshletsPerMesh;
    uint32_t maxDrawCount;
    uint32_t pad;
};

// Render UBO
struct RenderUBO
{
    glm::mat4 view;
    glm::mat4 proj;
    uint32_t debugMeshlet; // 0 = 正常, 1 = meshlet 调试着色
    uint32_t pad0, pad1, pad2;
};

// VkDrawIndexedIndirectCommand (20 bytes)
struct DrawIndexedIndirectCommand
{
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

// ════════════════ 辅助函数声明 ════════════════

static std::vector<uint32_t> readSPIRV(const std::string &filename);
static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeFilter, VkMemoryPropertyFlags props);
static void createBuffer(VkDevice device, VkPhysicalDevice physDev, VkDeviceSize size,
                         VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                         VkBuffer &buffer, VkDeviceMemory &memory);

// 提取视锥体 6 个平面 (Gribb/Hartmann 方法)
static void extractFrustumPlanes(const glm::mat4 &vp, glm::vec4 planes[6])
{
    planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]);
    planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);
    for (int i = 0; i < 6; i++)
    {
        float len = glm::length(glm::vec3(planes[i]));
        planes[i] /= len;
    }
}

// ════════════════════════════════════════════════════
// Meshlet 切分算法 (CPU 侧)
// ════════════════════════════════════════════════════
//
// 算法思路 (简化的贪心切分):
//   1. 遍历 mesh 的所有三角形
//   2. 逐三角形加入当前 meshlet
//   3. 统计当前 meshlet 引用的唯一顶点数
//   4. 当达到限制 (64 vert 或 126 tri) 时, 封闭当前 meshlet
//   5. 为每个 meshlet 计算 bounding sphere
//
// 注意: 这是简化的顺序切分, 适合面试/教学理解。
// 真实引擎中使用:
//   - meshoptimizer 库 (空间局部性优化)
//   - METIS 图分区 (缓存命中率最大化)
//   - 或 DirectX Mesh Shader 示例中的算法

struct MeshletBuildResult
{
    std::vector<GPUMeshlet> meshlets; // meshlet 描述符数组
    std::vector<uint32_t> indices;    // 重排后的索引缓冲 (所有 meshlet 的索引连续存放)
};

static MeshletBuildResult buildMeshlets(
    const std::vector<Vertex> &vertices,
    const std::vector<uint32_t> &srcIndices)
{
    MeshletBuildResult result;

    uint32_t triCount = (uint32_t)(srcIndices.size() / 3);
    uint32_t triIdx = 0;

    while (triIdx < triCount)
    {
        // 开始一个新 meshlet
        std::vector<uint32_t> meshletIndices; // 该 meshlet 的局部索引列表
        std::vector<uint32_t> uniqueVertices; // 唯一顶点集合
        auto findOrAdd = [&](uint32_t vi) -> bool
        {
            // 检查是否已在 uniqueVertices 中
            for (auto v : uniqueVertices)
            {
                if (v == vi)
                    return true; // 已存在
            }
            // 新顶点: 检查是否超出限制
            if (uniqueVertices.size() >= MAX_MESHLET_VERTICES)
            {
                return false; // 无法再加入
            }
            uniqueVertices.push_back(vi);
            return true;
        };

        uint32_t meshletTriCount = 0;

        while (triIdx < triCount && meshletTriCount < MAX_MESHLET_TRIANGLES)
        {
            uint32_t i0 = srcIndices[triIdx * 3 + 0];
            uint32_t i1 = srcIndices[triIdx * 3 + 1];
            uint32_t i2 = srcIndices[triIdx * 3 + 2];

            // 尝试添加 3 个顶点
            // 先检查是否会超出顶点限制
            auto tempUnique = uniqueVertices;
            auto testAdd = [&](uint32_t vi) -> bool
            {
                for (auto v : tempUnique)
                {
                    if (v == vi)
                        return true;
                }
                if (tempUnique.size() >= MAX_MESHLET_VERTICES)
                    return false;
                tempUnique.push_back(vi);
                return true;
            };

            if (!testAdd(i0) || !testAdd(i1) || !testAdd(i2))
            {
                break; // 该三角形无法放入当前 meshlet
            }

            // 确认添加
            findOrAdd(i0);
            findOrAdd(i1);
            findOrAdd(i2);
            meshletIndices.push_back(i0);
            meshletIndices.push_back(i1);
            meshletIndices.push_back(i2);
            meshletTriCount++;
            triIdx++;
        }

        if (meshletIndices.empty())
            break;

        // ─── 计算 bounding sphere ───
        // 简单方法: 先求 AABB 中心, 再算最大距离作为半径
        glm::vec3 minPos(FLT_MAX), maxPos(-FLT_MAX);
        for (auto vi : uniqueVertices)
        {
            glm::vec3 p = vertices[vi].position;
            minPos = glm::min(minPos, p);
            maxPos = glm::max(maxPos, p);
        }
        glm::vec3 center = (minPos + maxPos) * 0.5f;
        float radius = 0.0f;
        for (auto vi : uniqueVertices)
        {
            float d = glm::length(vertices[vi].position - center);
            radius = std::max(radius, d);
        }

        // ─── 创建 GPUMeshlet ───
        GPUMeshlet ml{};
        ml.indexOffset = (uint32_t)result.indices.size();
        ml.indexCount = (uint32_t)meshletIndices.size();
        ml.boundingSphere = glm::vec4(center, radius);

        result.meshlets.push_back(ml);

        // 追加索引到全局索引缓冲
        for (auto idx : meshletIndices)
        {
            result.indices.push_back(idx);
        }
    }

    return result;
}

// ════════════════ 生成立方体网格 ════════════════

static void generateCubeMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
{
    vertices.clear();
    indices.clear();

    struct Face
    {
        glm::vec3 positions[4];
        glm::vec3 normal;
    };

    Face faces[6] = {
        // 前面 (+Z)
        {{{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}}, {0, 0, 1}},
        // 后面 (-Z)
        {{{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}}, {0, 0, -1}},
        // 上面 (+Y)
        {{{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}}, {0, 1, 0}},
        // 下面 (-Y)
        {{{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}}, {0, -1, 0}},
        // 右面 (+X)
        {{{0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}}, {1, 0, 0}},
        // 左面 (-X)
        {{{-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}}, {-1, 0, 0}},
    };

    for (int f = 0; f < 6; f++)
    {
        uint32_t base = (uint32_t)vertices.size();
        for (int v = 0; v < 4; v++)
        {
            vertices.push_back({faces[f].positions[v], faces[f].normal});
        }
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
}

// ════════════════ 生成更复杂的球体网格 ════════════════
// 用 UV sphere 生成更多三角形, 以演示 meshlet 切分效果
// segments=16, rings=12 → 192 个四边形 = 384 个三角形 ≈ 3-4 个 meshlet

static void generateSphereMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices,
                               uint32_t segments = 16, uint32_t rings = 12)
{
    vertices.clear();
    indices.clear();

    const float PI = 3.14159265358979323846f;

    // 生成顶点
    for (uint32_t r = 0; r <= rings; r++)
    {
        float phi = PI * float(r) / float(rings); // 0 → π
        for (uint32_t s = 0; s <= segments; s++)
        {
            float theta = 2.0f * PI * float(s) / float(segments); // 0 → 2π

            float x = sinf(phi) * cosf(theta);
            float y = cosf(phi);
            float z = sinf(phi) * sinf(theta);

            Vertex v;
            v.position = glm::vec3(x, y, z) * 0.5f; // 半径 0.5 (与立方体尺寸一致)
            v.normal = glm::vec3(x, y, z);
            vertices.push_back(v);
        }
    }

    // 生成索引 (三角形)
    for (uint32_t r = 0; r < rings; r++)
    {
        for (uint32_t s = 0; s < segments; s++)
        {
            uint32_t tl = r * (segments + 1) + s;
            uint32_t tr = tl + 1;
            uint32_t bl = (r + 1) * (segments + 1) + s;
            uint32_t br = bl + 1;

            // 上三角形
            indices.push_back(tl);
            indices.push_back(bl);
            indices.push_back(tr);

            // 下三角形
            indices.push_back(tr);
            indices.push_back(bl);
            indices.push_back(br);
        }
    }
}

// ════════════════ 生成随机实例 ════════════════

static std::vector<InstanceData> generateInstances()
{
    std::vector<InstanceData> instances(INSTANCE_COUNT);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posDist(-20.0f, 20.0f);
    std::uniform_real_distribution<float> scaleDist(0.5f, 1.5f);
    std::uniform_real_distribution<float> colorDist(0.2f, 1.0f);

    for (uint32_t i = 0; i < INSTANCE_COUNT; i++)
    {
        instances[i].positionScale = glm::vec4(posDist(rng), posDist(rng), posDist(rng), scaleDist(rng));
        instances[i].color = glm::vec4(colorDist(rng), colorDist(rng), colorDist(rng), 1.0f);
    }
    return instances;
}

// ════════════════ MAIN ════════════════

int main()
{
    std::cout << "=== GPU Driven Meshlet Rendering Demo ===" << std::endl;
    std::cout << "实例数: " << INSTANCE_COUNT << std::endl;

    // ─── GLFW 初始化 ───
    if (!glfwInit())
    {
        std::cerr << "GLFW 初始化失败" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                                          "GPU Driven Meshlet Demo", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "窗口创建失败" << std::endl;
        glfwTerminate();
        return 1;
    }

    // 用户控制状态
    struct AppState
    {
        int cullMode = 0; // 0=atomicAdd, 1=prefix sum
        bool debugMeshlet = false;
    };
    AppState appState;
    glfwSetWindowUserPointer(window, &appState);
    glfwSetKeyCallback(window, [](GLFWwindow *w, int key, int, int action, int)
                       {
        if (action != GLFW_PRESS) return;
        auto *s = (AppState*)glfwGetWindowUserPointer(w);
        if (key == GLFW_KEY_1) { s->cullMode = 0; std::cout << "[切换] atomicAdd 模式" << std::endl; }
        if (key == GLFW_KEY_2) { s->cullMode = 1; std::cout << "[切换] Prefix Sum 模式" << std::endl; }
        if (key == GLFW_KEY_3) { s->debugMeshlet = !s->debugMeshlet;
            std::cout << "[切换] Meshlet 调试着色: " << (s->debugMeshlet ? "ON" : "OFF") << std::endl; }
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE); });

    std::cout << "控制: [1] atomicAdd  [2] Prefix Sum  [3] Meshlet着色  [ESC] 退出" << std::endl;

    // ─── Vulkan 实例 ───
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Meshlet Demo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    uint32_t glfwExtCount = 0;
    const char **glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char *> instExts(glfwExts, glfwExts + glfwExtCount);
    instExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkInstanceCreateInfo instCI{};
    instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;
    instCI.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instCI.enabledExtensionCount = (uint32_t)instExts.size();
    instCI.ppEnabledExtensionNames = instExts.data();

    VkInstance instance;
    if (vkCreateInstance(&instCI, nullptr, &instance) != VK_SUCCESS)
    {
        std::cerr << "Vulkan 实例创建失败" << std::endl;
        return 1;
    }

    // ─── Surface ───
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
    {
        std::cerr << "Surface 创建失败" << std::endl;
        return 1;
    }

    // ─── 物理设备 ───
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> physDevs(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, physDevs.data());
    VkPhysicalDevice physicalDevice = physDevs[0];

    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &devProps);
    std::cout << "GPU: " << devProps.deviceName << std::endl;

    // ─── 队列族 ───
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfProps.data());

    int queueFamily = -1;
    for (uint32_t i = 0; i < qfCount; i++)
    {
        if ((qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &present);
            if (present)
            {
                queueFamily = i;
                break;
            }
        }
    }
    if (queueFamily < 0)
    {
        std::cerr << "未找到 Graphics+Compute+Present 队列族" << std::endl;
        return 1;
    }

    // ─── 逻辑设备 ───
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = queueFamily;
    queueCI.queueCount = 1;
    queueCI.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures features{};
    // 查询 multiDrawIndirect 支持
    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);
    bool hasMultiDrawIndirect = supportedFeatures.multiDrawIndirect;
    if (hasMultiDrawIndirect)
    {
        features.multiDrawIndirect = VK_TRUE;
        std::cout << "multiDrawIndirect: 支持" << std::endl;
    }
    else
    {
        std::cout << "multiDrawIndirect: 不支持, 将使用逐个 draw 回退" << std::endl;
    }

    std::vector<const char *> devExts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset"};

    VkDeviceCreateInfo devCI{};
    devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &queueCI;
    devCI.pEnabledFeatures = &features;
    devCI.enabledExtensionCount = (uint32_t)devExts.size();
    devCI.ppEnabledExtensionNames = devExts.data();

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &devCI, nullptr, &device) != VK_SUCCESS)
    {
        std::cerr << "逻辑设备创建失败" << std::endl;
        return 1;
    }

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    // ─── Swapchain ───
    VkSurfaceCapabilitiesKHR surfCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfCaps);

    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfFormats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, surfFormats.data());

    VkSurfaceFormatKHR surfFmt = surfFormats[0];
    for (auto &f : surfFormats)
    {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            surfFmt = f;
            break;
        }
    }

    VkExtent2D swapExtent = surfCaps.currentExtent;
    if (swapExtent.width == UINT32_MAX)
        swapExtent = {WINDOW_WIDTH, WINDOW_HEIGHT};

    uint32_t imageCount = std::max(2u, surfCaps.minImageCount);
    if (surfCaps.maxImageCount > 0)
        imageCount = std::min(imageCount, surfCaps.maxImageCount);

    VkSwapchainCreateInfoKHR swapCI{};
    swapCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapCI.surface = surface;
    swapCI.minImageCount = imageCount;
    swapCI.imageFormat = surfFmt.format;
    swapCI.imageColorSpace = surfFmt.colorSpace;
    swapCI.imageExtent = swapExtent;
    swapCI.imageArrayLayers = 1;
    swapCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapCI.preTransform = surfCaps.currentTransform;
    swapCI.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapCI.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapCI.clipped = VK_TRUE;

    VkSwapchainKHR swapchain;
    if (vkCreateSwapchainKHR(device, &swapCI, nullptr, &swapchain) != VK_SUCCESS)
    {
        std::cerr << "Swapchain 创建失败" << std::endl;
        return 1;
    }

    uint32_t swapImageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &swapImageCount, nullptr);
    std::vector<VkImage> swapImages(swapImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &swapImageCount, swapImages.data());

    std::vector<VkImageView> swapImageViews(swapImageCount);
    for (uint32_t i = 0; i < swapImageCount; i++)
    {
        VkImageViewCreateInfo ivCI{};
        ivCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivCI.image = swapImages[i];
        ivCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivCI.format = surfFmt.format;
        ivCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device, &ivCI, nullptr, &swapImageViews[i]);
    }

    // ─── 深度缓冲 ───
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo depthImgCI{};
    depthImgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthImgCI.imageType = VK_IMAGE_TYPE_2D;
    depthImgCI.format = depthFormat;
    depthImgCI.extent = {swapExtent.width, swapExtent.height, 1};
    depthImgCI.mipLevels = 1;
    depthImgCI.arrayLayers = 1;
    depthImgCI.samples = VK_SAMPLE_COUNT_1_BIT;
    depthImgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImgCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImage depthImage;
    vkCreateImage(device, &depthImgCI, nullptr, &depthImage);

    VkMemoryRequirements depthMemReq;
    vkGetImageMemoryRequirements(device, depthImage, &depthMemReq);

    VkMemoryAllocateInfo depthAllocInfo{};
    depthAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    depthAllocInfo.allocationSize = depthMemReq.size;
    depthAllocInfo.memoryTypeIndex = findMemoryType(physicalDevice, depthMemReq.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory depthMemory;
    vkAllocateMemory(device, &depthAllocInfo, nullptr, &depthMemory);
    vkBindImageMemory(device, depthImage, depthMemory, 0);

    VkImageViewCreateInfo depthViewCI{};
    depthViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthViewCI.image = depthImage;
    depthViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewCI.format = depthFormat;
    depthViewCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkImageView depthView;
    vkCreateImageView(device, &depthViewCI, nullptr, &depthView);

    // ─── Render Pass ───
    std::array<VkAttachmentDescription, 2> attachments{};
    attachments[0].format = surfFmt.format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    attachments[1].format = depthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpCI{};
    rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = (uint32_t)attachments.size();
    rpCI.pAttachments = attachments.data();
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies = &dependency;

    VkRenderPass renderPass;
    vkCreateRenderPass(device, &rpCI, nullptr, &renderPass);

    // ─── Framebuffers ───
    std::vector<VkFramebuffer> framebuffers(swapImageCount);
    for (uint32_t i = 0; i < swapImageCount; i++)
    {
        std::array<VkImageView, 2> fbAttachments = {swapImageViews[i], depthView};
        VkFramebufferCreateInfo fbCI{};
        fbCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass = renderPass;
        fbCI.attachmentCount = (uint32_t)fbAttachments.size();
        fbCI.pAttachments = fbAttachments.data();
        fbCI.width = swapExtent.width;
        fbCI.height = swapExtent.height;
        fbCI.layers = 1;
        vkCreateFramebuffer(device, &fbCI, nullptr, &framebuffers[i]);
    }

    // ─── Command Pool & Buffer ───
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = queueFamily;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool commandPool;
    vkCreateCommandPool(device, &poolCI, nullptr, &commandPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

    // ════════════════ 数据准备 ════════════════

    // 生成球体网格 (比立方体更多三角形, 更好展示 meshlet)
    std::vector<Vertex> meshVertices;
    std::vector<uint32_t> meshIndices;
    generateSphereMesh(meshVertices, meshIndices, 16, 12);

    std::cout << "网格: " << meshVertices.size() << " 顶点, "
              << meshIndices.size() / 3 << " 三角形" << std::endl;

    // ─── Meshlet 切分 ───
    MeshletBuildResult meshletResult = buildMeshlets(meshVertices, meshIndices);
    uint32_t meshletCount = (uint32_t)meshletResult.meshlets.size();

    std::cout << "Meshlet 切分结果: " << meshletCount << " 个 meshlet" << std::endl;
    for (uint32_t i = 0; i < meshletCount; i++)
    {
        auto &ml = meshletResult.meshlets[i];
        std::cout << "  meshlet[" << i << "]: "
                  << ml.indexCount / 3 << " tri, "
                  << "offset=" << ml.indexOffset
                  << ", bs=(" << ml.boundingSphere.x << ","
                  << ml.boundingSphere.y << ","
                  << ml.boundingSphere.z << ") r="
                  << ml.boundingSphere.w << std::endl;
    }

    // 实例数据
    std::vector<InstanceData> instances = generateInstances();

    // 最大绘制命令数 = 实例数 × meshlet 数
    uint32_t maxDrawCount = INSTANCE_COUNT * meshletCount;

    // ─── GPU 缓冲区 ───

    // 顶点缓冲
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexMemory;
    VkDeviceSize vbSize = sizeof(Vertex) * meshVertices.size();
    createBuffer(device, physicalDevice, vbSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vertexBuffer, vertexMemory);
    {
        void *d;
        vkMapMemory(device, vertexMemory, 0, vbSize, 0, &d);
        memcpy(d, meshVertices.data(), vbSize);
        vkUnmapMemory(device, vertexMemory);
    }

    // 索引缓冲 (使用 meshlet 重排后的索引)
    VkBuffer indexBuffer;
    VkDeviceMemory indexMemory;
    VkDeviceSize ibSize = sizeof(uint32_t) * meshletResult.indices.size();
    createBuffer(device, physicalDevice, ibSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer, indexMemory);
    {
        void *d;
        vkMapMemory(device, indexMemory, 0, ibSize, 0, &d);
        memcpy(d, meshletResult.indices.data(), ibSize);
        vkUnmapMemory(device, indexMemory);
    }

    // 实例 SSBO
    VkBuffer instanceBuffer;
    VkDeviceMemory instanceMemory;
    VkDeviceSize instBufSize = sizeof(InstanceData) * INSTANCE_COUNT;
    createBuffer(device, physicalDevice, instBufSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 instanceBuffer, instanceMemory);
    {
        void *d;
        vkMapMemory(device, instanceMemory, 0, instBufSize, 0, &d);
        memcpy(d, instances.data(), instBufSize);
        vkUnmapMemory(device, instanceMemory);
    }

    // Meshlet SSBO
    VkBuffer meshletBuffer;
    VkDeviceMemory meshletMemory;
    VkDeviceSize mlBufSize = sizeof(GPUMeshlet) * meshletCount;
    createBuffer(device, physicalDevice, mlBufSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 meshletBuffer, meshletMemory);
    {
        void *d;
        vkMapMemory(device, meshletMemory, 0, mlBufSize, 0, &d);
        memcpy(d, meshletResult.meshlets.data(), mlBufSize);
        vkUnmapMemory(device, meshletMemory);
    }

    // Indirect Draw Command Buffer (VkDrawIndexedIndirectCommand × maxDrawCount)
    VkBuffer drawCmdBuffer;
    VkDeviceMemory drawCmdMemory;
    VkDeviceSize drawCmdBufSize = sizeof(DrawIndexedIndirectCommand) * maxDrawCount;
    createBuffer(device, physicalDevice, drawCmdBufSize,
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 drawCmdBuffer, drawCmdMemory);

    // Draw Count Buffer (单个 uint32_t: 可见 draw 数量)
    VkBuffer drawCountBuffer;
    VkDeviceMemory drawCountMemory;
    VkDeviceSize drawCountBufSize = sizeof(uint32_t);
    createBuffer(device, physicalDevice, drawCountBufSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 drawCountBuffer, drawCountMemory);

    // Cull UBO
    VkBuffer cullUboBuffer;
    VkDeviceMemory cullUboMemory;
    createBuffer(device, physicalDevice, sizeof(CullUBO),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 cullUboBuffer, cullUboMemory);

    // Render UBO
    VkBuffer renderUboBuffer;
    VkDeviceMemory renderUboMemory;
    createBuffer(device, physicalDevice, sizeof(RenderUBO),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 renderUboBuffer, renderUboMemory);

    std::cout << "GPU 缓冲区创建完成" << std::endl;

    // ════════════════ Descriptor Set Layout ════════════════

    // Compute: 5 bindings
    //   0: CullUBO (uniform)
    //   1: InstanceBuffer (storage, read)
    //   2: MeshletBuffer (storage, read)
    //   3: DrawCmdBuffer (storage, rw)
    //   4: DrawCountBuffer (storage, rw)
    std::array<VkDescriptorSetLayoutBinding, 5> compBindings{};
    compBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo compDSLCI{};
    compDSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    compDSLCI.bindingCount = (uint32_t)compBindings.size();
    compDSLCI.pBindings = compBindings.data();

    VkDescriptorSetLayout compDSLayout;
    vkCreateDescriptorSetLayout(device, &compDSLCI, nullptr, &compDSLayout);

    // Graphics: 2 bindings
    //   0: RenderUBO (uniform, vertex)
    //   1: InstanceBuffer (storage, vertex)
    std::array<VkDescriptorSetLayoutBinding, 2> gfxBindings{};
    gfxBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    gfxBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo gfxDSLCI{};
    gfxDSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gfxDSLCI.bindingCount = (uint32_t)gfxBindings.size();
    gfxDSLCI.pBindings = gfxBindings.data();

    VkDescriptorSetLayout gfxDSLayout;
    vkCreateDescriptorSetLayout(device, &gfxDSLCI, nullptr, &gfxDSLayout);

    // ─── Descriptor Pool ───
    std::array<VkDescriptorPoolSize, 2> dpSizes{};
    dpSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4};
    dpSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12};

    VkDescriptorPoolCreateInfo dpCI{};
    dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets = 4;
    dpCI.poolSizeCount = (uint32_t)dpSizes.size();
    dpCI.pPoolSizes = dpSizes.data();

    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &dpCI, nullptr, &descriptorPool);

    // ─── 分配 Descriptor Sets ───
    VkDescriptorSetAllocateInfo compDSAI{};
    compDSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    compDSAI.descriptorPool = descriptorPool;
    compDSAI.descriptorSetCount = 1;
    compDSAI.pSetLayouts = &compDSLayout;

    VkDescriptorSet compDescSet;
    vkAllocateDescriptorSets(device, &compDSAI, &compDescSet);

    VkDescriptorSetAllocateInfo gfxDSAI{};
    gfxDSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    gfxDSAI.descriptorPool = descriptorPool;
    gfxDSAI.descriptorSetCount = 1;
    gfxDSAI.pSetLayouts = &gfxDSLayout;

    VkDescriptorSet gfxDescSet;
    vkAllocateDescriptorSets(device, &gfxDSAI, &gfxDescSet);

    // ─── 更新 Descriptor Sets ───
    // Compute set (5 bindings)
    VkDescriptorBufferInfo compUboInfo{cullUboBuffer, 0, sizeof(CullUBO)};
    VkDescriptorBufferInfo compInstInfo{instanceBuffer, 0, instBufSize};
    VkDescriptorBufferInfo compMlInfo{meshletBuffer, 0, mlBufSize};
    VkDescriptorBufferInfo compDrawInfo{drawCmdBuffer, 0, drawCmdBufSize};
    VkDescriptorBufferInfo compCountInfo{drawCountBuffer, 0, drawCountBufSize};

    std::array<VkWriteDescriptorSet, 5> compWrites{};
    compWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &compUboInfo, nullptr};
    compWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compInstInfo, nullptr};
    compWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compMlInfo, nullptr};
    compWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 3, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compDrawInfo, nullptr};
    compWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 4, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compCountInfo, nullptr};
    vkUpdateDescriptorSets(device, (uint32_t)compWrites.size(), compWrites.data(), 0, nullptr);

    // Graphics set (2 bindings)
    VkDescriptorBufferInfo gfxUboInfo{renderUboBuffer, 0, sizeof(RenderUBO)};
    VkDescriptorBufferInfo gfxInstInfo{instanceBuffer, 0, instBufSize};

    std::array<VkWriteDescriptorSet, 2> gfxWrites{};
    gfxWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gfxDescSet, 0, 0, 1,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &gfxUboInfo, nullptr};
    gfxWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gfxDescSet, 1, 0, 1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gfxInstInfo, nullptr};
    vkUpdateDescriptorSets(device, (uint32_t)gfxWrites.size(), gfxWrites.data(), 0, nullptr);

    // ════════════════ Pipeline Layout ════════════════

    VkPipelineLayoutCreateInfo compPLCI{};
    compPLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compPLCI.setLayoutCount = 1;
    compPLCI.pSetLayouts = &compDSLayout;

    VkPipelineLayout compPipelineLayout;
    vkCreatePipelineLayout(device, &compPLCI, nullptr, &compPipelineLayout);

    VkPipelineLayoutCreateInfo gfxPLCI{};
    gfxPLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfxPLCI.setLayoutCount = 1;
    gfxPLCI.pSetLayouts = &gfxDSLayout;

    VkPipelineLayout gfxPipelineLayout;
    vkCreatePipelineLayout(device, &gfxPLCI, nullptr, &gfxPipelineLayout);

    // ════════════════ Shader Modules ════════════════

    auto loadShaderModule = [&](const std::string &path) -> VkShaderModule
    {
        auto code = readSPIRV(path);
        VkShaderModuleCreateInfo smCI{};
        smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smCI.codeSize = code.size() * sizeof(uint32_t);
        smCI.pCode = code.data();
        VkShaderModule sm;
        if (vkCreateShaderModule(device, &smCI, nullptr, &sm) != VK_SUCCESS)
        {
            std::cerr << "Shader module 创建失败: " << path << std::endl;
            exit(1);
        }
        return sm;
    };

    std::string shaderDir = SHADER_DIR;
    VkShaderModule cullAtomicSM = loadShaderModule(shaderDir + "meshlet_cull_atomic.comp.spv");
    VkShaderModule cullScanSM = loadShaderModule(shaderDir + "meshlet_cull_scan.comp.spv");
    VkShaderModule vertSM = loadShaderModule(shaderDir + "meshlet.vert.spv");
    VkShaderModule fragSM = loadShaderModule(shaderDir + "meshlet.frag.spv");
    std::cout << "Shader 加载成功" << std::endl;

    // ════════════════ Compute Pipelines ════════════════

    auto createComputePipeline = [&](VkShaderModule sm) -> VkPipeline
    {
        VkPipelineShaderStageCreateInfo stageCI{};
        stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageCI.module = sm;
        stageCI.pName = "main";

        VkComputePipelineCreateInfo cpCI{};
        cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpCI.stage = stageCI;
        cpCI.layout = compPipelineLayout;

        VkPipeline pipeline;
        vkCreateComputePipelines(device, nullptr, 1, &cpCI, nullptr, &pipeline);
        return pipeline;
    };

    VkPipeline cullAtomicPipeline = createComputePipeline(cullAtomicSM);
    VkPipeline cullScanPipeline = createComputePipeline(cullScanSM);

    // ════════════════ Graphics Pipeline ════════════════

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertSM;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragSM;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo gfxStages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription vbBind{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 2> vbAttrs{};
    vbAttrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    vbAttrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};

    VkPipelineVertexInputStateCreateInfo viCI{};
    viCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viCI.vertexBindingDescriptionCount = 1;
    viCI.pVertexBindingDescriptions = &vbBind;
    viCI.vertexAttributeDescriptionCount = (uint32_t)vbAttrs.size();
    viCI.pVertexAttributeDescriptions = vbAttrs.data();

    VkPipelineInputAssemblyStateCreateInfo iaCI{};
    iaCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)swapExtent.width, (float)swapExtent.height, 0, 1};
    VkRect2D scissor{{0, 0}, swapExtent};

    VkPipelineViewportStateCreateInfo vpCI{};
    vpCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpCI.viewportCount = 1;
    vpCI.pViewports = &vp;
    vpCI.scissorCount = 1;
    vpCI.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rsCI{};
    rsCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsCI.polygonMode = VK_POLYGON_MODE_FILL;
    rsCI.lineWidth = 1.0f;
    rsCI.cullMode = VK_CULL_MODE_BACK_BIT;
    rsCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo msCI{};
    msCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsCI{};
    dsCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsCI.depthTestEnable = VK_TRUE;
    dsCI.depthWriteEnable = VK_TRUE;
    dsCI.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cbCI{};
    cbCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbCI.attachmentCount = 1;
    cbCI.pAttachments = &cbAtt;

    VkGraphicsPipelineCreateInfo gpCI{};
    gpCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpCI.stageCount = 2;
    gpCI.pStages = gfxStages;
    gpCI.pVertexInputState = &viCI;
    gpCI.pInputAssemblyState = &iaCI;
    gpCI.pViewportState = &vpCI;
    gpCI.pRasterizationState = &rsCI;
    gpCI.pMultisampleState = &msCI;
    gpCI.pDepthStencilState = &dsCI;
    gpCI.pColorBlendState = &cbCI;
    gpCI.layout = gfxPipelineLayout;
    gpCI.renderPass = renderPass;
    gpCI.subpass = 0;

    VkPipeline graphicsPipeline;
    if (vkCreateGraphicsPipelines(device, nullptr, 1, &gpCI, nullptr, &graphicsPipeline) != VK_SUCCESS)
    {
        std::cerr << "图形管线创建失败" << std::endl;
        return 1;
    }
    std::cout << "管线创建完成" << std::endl;

    vkDestroyShaderModule(device, cullAtomicSM, nullptr);
    vkDestroyShaderModule(device, cullScanSM, nullptr);
    vkDestroyShaderModule(device, vertSM, nullptr);
    vkDestroyShaderModule(device, fragSM, nullptr);

    // ─── 同步原语 ───
    VkSemaphoreCreateInfo semCI{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenCI{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};

    VkSemaphore imageAvailSem, renderDoneSem;
    VkFence inFlightFence;
    vkCreateSemaphore(device, &semCI, nullptr, &imageAvailSem);
    vkCreateSemaphore(device, &semCI, nullptr, &renderDoneSem);
    vkCreateFence(device, &fenCI, nullptr, &inFlightFence);

    // ════════════════ 主循环 ════════════════

    std::cout << "\n=== 渲染循环开始 ===" << std::endl;
    std::cout << "总 meshlet-instance 数: " << INSTANCE_COUNT << " × " << meshletCount
              << " = " << maxDrawCount << std::endl;

    float time = 0.0f;
    uint32_t frameCount = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        time += 0.016f;

        // ─── 相机绕原点旋转 ───
        float camRadius = 35.0f;
        float camAngle = time * 0.5f;
        glm::vec3 camPos(camRadius * cosf(camAngle), 15.0f, camRadius * sinf(camAngle));
        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0), glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                                          (float)swapExtent.width / (float)swapExtent.height, 0.1f, 100.0f);
        proj[1][1] *= -1.0f; // Vulkan Y flip

        glm::mat4 viewProj = proj * view;

        // ─── 更新 Cull UBO ───
        CullUBO cullUbo{};
        cullUbo.viewProj = viewProj;
        extractFrustumPlanes(viewProj, cullUbo.frustumPlanes);
        cullUbo.instanceCount = INSTANCE_COUNT;
        cullUbo.meshletsPerMesh = meshletCount;
        cullUbo.maxDrawCount = maxDrawCount;
        {
            void *d;
            vkMapMemory(device, cullUboMemory, 0, sizeof(CullUBO), 0, &d);
            memcpy(d, &cullUbo, sizeof(CullUBO));
            vkUnmapMemory(device, cullUboMemory);
        }

        // ─── 更新 Render UBO ───
        RenderUBO renderUbo{};
        renderUbo.view = view;
        renderUbo.proj = proj;
        renderUbo.debugMeshlet = appState.debugMeshlet ? 1 : 0;
        {
            void *d;
            vkMapMemory(device, renderUboMemory, 0, sizeof(RenderUBO), 0, &d);
            memcpy(d, &renderUbo, sizeof(RenderUBO));
            vkUnmapMemory(device, renderUboMemory);
        }

        // ─── 等待上一帧 ───
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailSem, nullptr, &imageIndex);

        // ─── 录制命令 ───
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginCI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginCI);

        // ═══ 阶段 0: 清零 DrawCmd 和 DrawCount Buffer ═══
        // 用 vkCmdFillBuffer 在 GPU 端清零, 确保未被 compute 写入的 draw
        // 命令的 instanceCount=0, GPU 会自动跳过这些空 draw
        vkCmdFillBuffer(commandBuffer, drawCmdBuffer, 0, drawCmdBufSize, 0);
        vkCmdFillBuffer(commandBuffer, drawCountBuffer, 0, sizeof(uint32_t), 0);

        // Barrier: Transfer → Compute (确保 fill 完成后再写入)
        VkMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &fillBarrier, 0, nullptr, 0, nullptr);

        // ═══ 阶段 1: Compute — Meshlet Frustum Culling ═══
        VkPipeline currentCullPipeline = (appState.cullMode == 0) ? cullAtomicPipeline : cullScanPipeline;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, currentCullPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compPipelineLayout, 0, 1, &compDescSet, 0, nullptr);

        // dispatch: 每个线程处理一个 (instance, meshlet)
        uint32_t totalMeshletInstances = INSTANCE_COUNT * meshletCount;
        uint32_t groupCount = (totalMeshletInstances + COMPUTE_LOCAL_SIZE - 1) / COMPUTE_LOCAL_SIZE;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        // ═══ Pipeline Barrier: Compute → Draw Indirect ═══
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        // ═══ 阶段 2: Graphics Render Pass ═══
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.05f, 0.05f, 0.1f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpBeginInfo{};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = renderPass;
        rpBeginInfo.framebuffer = framebuffers[imageIndex];
        rpBeginInfo.renderArea = {{0, 0}, swapExtent};
        rpBeginInfo.clearValueCount = (uint32_t)clearValues.size();
        rpBeginInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                gfxPipelineLayout, 0, 1, &gfxDescSet, 0, nullptr);

        VkBuffer vbs[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vbs, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // ═══ Meshlet GPU Driven: Multi Draw Indirect ═══
        // 发出 maxDrawCount 个 draw, 但只有 compute 写入的前 drawCount 个有效
        // 其余 instanceCount=0 (被 vkCmdFillBuffer 清零), GPU 自动跳过
        if (hasMultiDrawIndirect)
        {
            // 单次调用发出所有间接绘制 (高效)
            vkCmdDrawIndexedIndirect(commandBuffer, drawCmdBuffer, 0,
                                     maxDrawCount, sizeof(DrawIndexedIndirectCommand));
        }
        else
        {
            // 逐个发出 indirect draw (兼容回退)
            for (uint32_t i = 0; i < maxDrawCount; i++)
            {
                vkCmdDrawIndexedIndirect(commandBuffer, drawCmdBuffer,
                                         i * sizeof(DrawIndexedIndirectCommand), 1, sizeof(DrawIndexedIndirectCommand));
            }
        }

        vkCmdEndRenderPass(commandBuffer);
        vkEndCommandBuffer(commandBuffer);

        // ─── 提交 ───
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailSem;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderDoneSem;

        vkQueueSubmit(queue, 1, &submitInfo, inFlightFence);

        // ─── Present ───
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderDoneSem;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(queue, &presentInfo);

        frameCount++;
        if (frameCount % 300 == 0)
        {
            // 读回 drawCount 供调试
            vkDeviceWaitIdle(device);
            uint32_t visibleDraws = 0;
            {
                void *d;
                vkMapMemory(device, drawCountMemory, 0, sizeof(uint32_t), 0, &d);
                visibleDraws = *(uint32_t *)d;
                vkUnmapMemory(device, drawCountMemory);
            }
            std::cout << "[帧 " << frameCount << "] 模式: "
                      << (appState.cullMode == 0 ? "atomicAdd" : "Prefix Sum")
                      << " | 可见 meshlet draws: " << visibleDraws
                      << " / " << maxDrawCount << std::endl;
        }
    }

    vkDeviceWaitIdle(device);
    std::cout << "\n=== 清理资源 ===" << std::endl;

    // ════════════════ 清理 ════════════════
    vkDestroySemaphore(device, imageAvailSem, nullptr);
    vkDestroySemaphore(device, renderDoneSem, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipeline(device, cullAtomicPipeline, nullptr);
    vkDestroyPipeline(device, cullScanPipeline, nullptr);
    vkDestroyPipelineLayout(device, gfxPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, compPipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, compDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, gfxDSLayout, nullptr);

    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    auto destroyBuf = [&](VkBuffer b, VkDeviceMemory m)
    {
        vkDestroyBuffer(device, b, nullptr);
        vkFreeMemory(device, m, nullptr);
    };
    destroyBuf(vertexBuffer, vertexMemory);
    destroyBuf(indexBuffer, indexMemory);
    destroyBuf(instanceBuffer, instanceMemory);
    destroyBuf(meshletBuffer, meshletMemory);
    destroyBuf(drawCmdBuffer, drawCmdMemory);
    destroyBuf(drawCountBuffer, drawCountMemory);
    destroyBuf(cullUboBuffer, cullUboMemory);
    destroyBuf(renderUboBuffer, renderUboMemory);

    vkDestroyImageView(device, depthView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthMemory, nullptr);

    for (auto iv : swapImageViews)
        vkDestroyImageView(device, iv, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "程序正常退出, 共渲染 " << frameCount << " 帧" << std::endl;
    return 0;
}

// ════════════════ 辅助函数实现 ════════════════

static std::vector<uint32_t> readSPIRV(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "无法打开 SPIR-V 文件: " << filename << std::endl;
        exit(1);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char *)buffer.data(), fileSize);
    return buffer;
}

static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeFilter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    std::cerr << "未找到合适的内存类型" << std::endl;
    exit(1);
}

static void createBuffer(VkDevice device, VkPhysicalDevice physDev, VkDeviceSize size,
                         VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                         VkBuffer &buffer, VkDeviceMemory &memory)
{
    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size = size;
    bufCI.usage = usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufCI, nullptr, &buffer) != VK_SUCCESS)
    {
        std::cerr << "Buffer 创建失败" << std::endl;
        exit(1);
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physDev, memReq.memoryTypeBits, props);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
        std::cerr << "Memory 分配失败" << std::endl;
        exit(1);
    }

    vkBindBufferMemory(device, buffer, memory, 0);
}

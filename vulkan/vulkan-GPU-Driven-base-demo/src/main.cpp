// ════════════════════════════════════════════════════════════
// GPU Driven Rendering Demo - 1000 个立方体实例化 + GPU Frustum Culling
// ════════════════════════════════════════════════════════════
//
// 整体流程:
//   1. CPU 生成 1000 个实例数据 (位置/颜色/缩放)
//   2. 每帧:
//      a. CPU 更新视锥体参数 (UBO)
//      b. GPU Compute Shader 做 frustum culling
//      c. Compute 写入 indirect draw buffer + 可见实例 ID 列表
//      d. Pipeline Barrier: compute → indirect draw
//      e. vkCmdDrawIndexedIndirect 使用 GPU 生成的 indirect buffer
//   3. CPU 不做任何 culling 计算
//
// 提供两种 culling 模式:
//   [1] atomicAdd: 每个可见实例用 atomicAdd 获取写入位置
//   [2] prefix sum: 工作组内做 exclusive scan，减少原子操作
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

// ════════════════ 常量定义 ════════════════

static const uint32_t WINDOW_WIDTH = 1024;
static const uint32_t WINDOW_HEIGHT = 768;
static const uint32_t INSTANCE_COUNT = 1000;   // 总实例数
static const uint32_t CUBE_VERTEX_COUNT = 24;  // 立方体顶点数 (每面4个 × 6面)
static const uint32_t CUBE_INDEX_COUNT = 36;   // 立方体索引数 (每面2三角形 × 6面)
static const uint32_t COMPUTE_LOCAL_SIZE = 64; // 对应 shader 中 local_size_x

// ════════════════ 数据结构 ════════════════

// 立方体顶点 (与 cube.vert 对应)
struct Vertex
{
    glm::vec3 position; // location = 0
    glm::vec3 normal;   // location = 1
};

// 实例数据 (与 shader 中 InstanceData 对应, std430 布局)
struct InstanceData
{
    glm::vec4 positionScale; // xyz = 世界坐标, w = 缩放
    glm::vec4 color;         // rgb = 颜色, a = 未使用
};

// Compute Shader UBO (与 shader 中 CullUBO 对应)
struct CullUBO
{
    glm::mat4 viewProj;         // 视图投影矩阵
    glm::vec4 frustumPlanes[6]; // 视锥体 6 个平面
    uint32_t instanceCount;     // 实例总数
    uint32_t pad0, pad1, pad2;  // 对齐填充
};

// Render UBO (与 cube.vert 中 RenderUBO 对应)
struct RenderUBO
{
    glm::mat4 view; // 视图矩阵
    glm::mat4 proj; // 投影矩阵
};

// VkDrawIndexedIndirectCommand 结构 (Vulkan 标准布局)
// ┌──────────────┬──────────────────┬────────────┬──────────────┬───────────────┐
// │ indexCount   │ instanceCount    │ firstIndex │ vertexOffset │ firstInstance │
// │ = 36         │ = visibleCount   │ = 0        │ = 0          │ = 0           │
// └──────────────┴──────────────────┴────────────┴──────────────┴───────────────┘

// ════════════════ 辅助函数声明 ════════════════

static std::vector<uint32_t> readSPIRV(const std::string &filename);
static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeFilter, VkMemoryPropertyFlags props);
static void createBuffer(VkDevice device, VkPhysicalDevice physDev, VkDeviceSize size,
                         VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                         VkBuffer &buffer, VkDeviceMemory &memory);

// 从 glm::mat4 投影视图矩阵中提取视锥体的6个平面
// Gribb/Hartmann 方法
static void extractFrustumPlanes(const glm::mat4 &vp, glm::vec4 planes[6])
{
    // 左平面:   row3 + row0
    planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0],
                          vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    // 右平面:   row3 - row0
    planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0],
                          vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    // 底平面:   row3 + row1
    planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1],
                          vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    // 顶平面:   row3 - row1
    planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1],
                          vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    // 近平面:   row3 + row2
    planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2],
                          vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]);
    // 远平面:   row3 - row2
    planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2],
                          vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);

    // 归一化每个平面: 使 (a,b,c) 成为单位向量
    for (int i = 0; i < 6; i++)
    {
        float len = glm::length(glm::vec3(planes[i]));
        planes[i] /= len;
    }
}

// ════════════════ 生成立方体网格 ════════════════

static void generateCubeMesh(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
{
    vertices.clear();
    indices.clear();

    // 6 个面，每面 4 个唯一顶点 (独立法线)
    struct Face
    {
        glm::vec3 positions[4];
        glm::vec3 normal;
    };

    // 单位立方体 [-0.5, 0.5]
    Face faces[6] = {
        // 前面 (+Z)
        {{{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
         {0.0f, 0.0f, 1.0f}},
        // 后面 (-Z)
        {{{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}},
         {0.0f, 0.0f, -1.0f}},
        // 上面 (+Y)
        {{{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}},
         {0.0f, 1.0f, 0.0f}},
        // 下面 (-Y)
        {{{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}},
         {0.0f, -1.0f, 0.0f}},
        // 右面 (+X)
        {{{0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}},
         {1.0f, 0.0f, 0.0f}},
        // 左面 (-X)
        {{{-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}},
         {-1.0f, 0.0f, 0.0f}},
    };

    for (int f = 0; f < 6; f++)
    {
        uint32_t base = static_cast<uint32_t>(vertices.size());
        for (int v = 0; v < 4; v++)
        {
            vertices.push_back({faces[f].positions[v], faces[f].normal});
        }
        // 两个三角形: 0-1-2, 0-2-3
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
}

// ════════════════ 生成 1000 个随机实例 ════════════════

static std::vector<InstanceData> generateInstances()
{
    std::vector<InstanceData> instances(INSTANCE_COUNT);
    std::mt19937 rng(42); // 固定种子，可复现
    std::uniform_real_distribution<float> posDist(-20.0f, 20.0f);
    std::uniform_real_distribution<float> scaleDist(0.3f, 1.0f);
    std::uniform_real_distribution<float> colorDist(0.2f, 1.0f);

    for (uint32_t i = 0; i < INSTANCE_COUNT; i++)
    {
        instances[i].positionScale = glm::vec4(
            posDist(rng), posDist(rng), posDist(rng), // 世界坐标
            scaleDist(rng)                            // 缩放
        );
        instances[i].color = glm::vec4(
            colorDist(rng), colorDist(rng), colorDist(rng), 1.0f);
    }
    return instances;
}

// ════════════════ MAIN ════════════════

int main()
{
    std::cout << "=== GPU Driven Rendering Demo ===" << std::endl;
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
                                          "GPU Driven Rendering - 1000 Cubes", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "窗口创建失败" << std::endl;
        glfwTerminate();
        return 1;
    }

    // 当前裁剪模式: 0=atomicAdd, 1=prefix sum
    int cullMode = 0;
    glfwSetWindowUserPointer(window, &cullMode);
    glfwSetKeyCallback(window, [](GLFWwindow *w, int key, int /*scancode*/, int action, int /*mods*/)
                       {
        if (action != GLFW_PRESS) return;
        int* mode = (int*)glfwGetWindowUserPointer(w);
        if (key == GLFW_KEY_1) { *mode = 0; std::cout << "[切换] atomicAdd 模式" << std::endl; }
        if (key == GLFW_KEY_2) { *mode = 1; std::cout << "[切换] Prefix Sum 模式" << std::endl; }
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE); });

    std::cout << "控制: [1] atomicAdd  [2] Prefix Sum  [ESC] 退出" << std::endl;

    // ─── Vulkan 实例 ───
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPU Driven Demo";
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
    std::cout << "Vulkan 实例创建成功" << std::endl;

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

    // ─── 队列族: 同时支持 Graphics + Compute ───
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
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
            if (presentSupport)
            {
                queueFamily = i;
                break;
            }
        }
    }
    if (queueFamily < 0)
    {
        std::cerr << "未找到同时支持 Graphics + Compute + Present 的队列族" << std::endl;
        return 1;
    }
    std::cout << "队列族: " << queueFamily << " (Graphics + Compute + Present)" << std::endl;

    // ─── 逻辑设备 ───
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = queueFamily;
    queueCI.queueCount = 1;
    queueCI.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures features{};
    // multiDrawIndirect 在 MoltenVK 上可能不可用，我们使用 drawIndirect 足够

    std::vector<const char *> devExts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset" // MoltenVK 需要
    };

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
    {
        swapExtent = {WINDOW_WIDTH, WINDOW_HEIGHT};
    }

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
    std::cout << "Swapchain 创建成功 (" << swapImageCount << " images, "
              << swapExtent.width << "x" << swapExtent.height << ")" << std::endl;

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
    // 颜色附件
    attachments[0].format = surfFmt.format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // 深度附件
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

    // 立方体网格
    std::vector<Vertex> cubeVertices;
    std::vector<uint32_t> cubeIndices;
    generateCubeMesh(cubeVertices, cubeIndices);
    assert(cubeVertices.size() == CUBE_VERTEX_COUNT);
    assert(cubeIndices.size() == CUBE_INDEX_COUNT);

    // 实例数据
    std::vector<InstanceData> instances = generateInstances();

    // ─── GPU 缓冲区创建 ───

    // 顶点缓冲
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexMemory;
    VkDeviceSize vbSize = sizeof(Vertex) * cubeVertices.size();
    createBuffer(device, physicalDevice, vbSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vertexBuffer, vertexMemory);
    {
        void *data;
        vkMapMemory(device, vertexMemory, 0, vbSize, 0, &data);
        memcpy(data, cubeVertices.data(), vbSize);
        vkUnmapMemory(device, vertexMemory);
    }

    // 索引缓冲
    VkBuffer indexBuffer;
    VkDeviceMemory indexMemory;
    VkDeviceSize ibSize = sizeof(uint32_t) * cubeIndices.size();
    createBuffer(device, physicalDevice, ibSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer, indexMemory);
    {
        void *data;
        vkMapMemory(device, indexMemory, 0, ibSize, 0, &data);
        memcpy(data, cubeIndices.data(), ibSize);
        vkUnmapMemory(device, indexMemory);
    }

    // 实例 SSBO (compute + vertex 共享)
    VkBuffer instanceBuffer;
    VkDeviceMemory instanceMemory;
    VkDeviceSize instBufSize = sizeof(InstanceData) * INSTANCE_COUNT;
    createBuffer(device, physicalDevice, instBufSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 instanceBuffer, instanceMemory);
    {
        void *data;
        vkMapMemory(device, instanceMemory, 0, instBufSize, 0, &data);
        memcpy(data, instances.data(), instBufSize);
        vkUnmapMemory(device, instanceMemory);
    }

    // Indirect Draw Buffer (VkDrawIndexedIndirectCommand = 5 × uint32_t = 20 bytes)
    // ┌──────────────┬────────────────┬────────────┬──────────────┬───────────────┐
    // │ indexCount(4) │instanceCount(4)│firstIndex(4)│vertexOffset(4)│firstInstance(4)│
    // └──────────────┴────────────────┴────────────┴──────────────┴───────────────┘
    VkBuffer indirectBuffer;
    VkDeviceMemory indirectMemory;
    VkDeviceSize indirectBufSize = sizeof(uint32_t) * 5;
    createBuffer(device, physicalDevice, indirectBufSize,
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indirectBuffer, indirectMemory);

    // Visible ID Buffer (可见实例索引列表)
    VkBuffer visibleBuffer;
    VkDeviceMemory visibleMemory;
    VkDeviceSize visBufSize = sizeof(uint32_t) * INSTANCE_COUNT;
    createBuffer(device, physicalDevice, visBufSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 visibleBuffer, visibleMemory);

    // Cull UBO
    VkBuffer cullUboBuffer;
    VkDeviceMemory cullUboMemory;
    createBuffer(device, physicalDevice, sizeof(CullUBO),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 cullUboBuffer, cullUboMemory);

    // Render UBO
    VkBuffer renderUboBuffer;
    VkDeviceMemory renderUboMemory;
    createBuffer(device, physicalDevice, sizeof(RenderUBO),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 renderUboBuffer, renderUboMemory);

    std::cout << "GPU 缓冲区创建完成" << std::endl;

    // ════════════════ Descriptor Set Layout ════════════════

    // Compute Descriptor Set Layout (set 0)
    // binding 0: CullUBO (uniform)
    // binding 1: InstanceBuffer (storage, read)
    // binding 2: IndirectBuffer (storage, read/write)
    // binding 3: VisibleBuffer (storage, read/write)
    std::array<VkDescriptorSetLayoutBinding, 4> compBindings{};
    compBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    compBindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo compDSLCI{};
    compDSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    compDSLCI.bindingCount = (uint32_t)compBindings.size();
    compDSLCI.pBindings = compBindings.data();

    VkDescriptorSetLayout compDSLayout;
    vkCreateDescriptorSetLayout(device, &compDSLCI, nullptr, &compDSLayout);

    // Graphics Descriptor Set Layout (set 0)
    // binding 0: RenderUBO (uniform, vertex)
    // binding 1: InstanceBuffer (storage, vertex)
    // binding 2: VisibleBuffer (storage, vertex)
    std::array<VkDescriptorSetLayoutBinding, 3> gfxBindings{};
    gfxBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    gfxBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    gfxBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo gfxDSLCI{};
    gfxDSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gfxDSLCI.bindingCount = (uint32_t)gfxBindings.size();
    gfxDSLCI.pBindings = gfxBindings.data();

    VkDescriptorSetLayout gfxDSLayout;
    vkCreateDescriptorSetLayout(device, &gfxDSLCI, nullptr, &gfxDSLayout);

    // ─── Descriptor Pool ───
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10};

    VkDescriptorPoolCreateInfo dpCI{};
    dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets = 4;
    dpCI.poolSizeCount = (uint32_t)poolSizes.size();
    dpCI.pPoolSizes = poolSizes.data();

    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &dpCI, nullptr, &descriptorPool);

    // ─── 分配 Descriptor Sets ───
    // compute descriptor set (用于 cull_atomic 和 cull_scan)
    VkDescriptorSetAllocateInfo compDSAI{};
    compDSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    compDSAI.descriptorPool = descriptorPool;
    compDSAI.descriptorSetCount = 1;
    compDSAI.pSetLayouts = &compDSLayout;

    VkDescriptorSet compDescSet;
    vkAllocateDescriptorSets(device, &compDSAI, &compDescSet);

    // graphics descriptor set
    VkDescriptorSetAllocateInfo gfxDSAI{};
    gfxDSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    gfxDSAI.descriptorPool = descriptorPool;
    gfxDSAI.descriptorSetCount = 1;
    gfxDSAI.pSetLayouts = &gfxDSLayout;

    VkDescriptorSet gfxDescSet;
    vkAllocateDescriptorSets(device, &gfxDSAI, &gfxDescSet);

    // ─── 更新 Descriptor Sets ───
    // Compute set
    VkDescriptorBufferInfo compUboInfo{cullUboBuffer, 0, sizeof(CullUBO)};
    VkDescriptorBufferInfo compInstInfo{instanceBuffer, 0, instBufSize};
    VkDescriptorBufferInfo compIndirectInfo{indirectBuffer, 0, indirectBufSize};
    VkDescriptorBufferInfo compVisInfo{visibleBuffer, 0, visBufSize};

    std::array<VkWriteDescriptorSet, 4> compWrites{};
    compWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &compUboInfo, nullptr};
    compWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compInstInfo, nullptr};
    compWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compIndirectInfo, nullptr};
    compWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 3, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compVisInfo, nullptr};
    vkUpdateDescriptorSets(device, (uint32_t)compWrites.size(), compWrites.data(), 0, nullptr);

    // Graphics set
    VkDescriptorBufferInfo gfxUboInfo{renderUboBuffer, 0, sizeof(RenderUBO)};
    VkDescriptorBufferInfo gfxInstInfo{instanceBuffer, 0, instBufSize};
    VkDescriptorBufferInfo gfxVisInfo{visibleBuffer, 0, visBufSize};

    std::array<VkWriteDescriptorSet, 3> gfxWrites{};
    gfxWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gfxDescSet, 0, 0, 1,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &gfxUboInfo, nullptr};
    gfxWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gfxDescSet, 1, 0, 1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gfxInstInfo, nullptr};
    gfxWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gfxDescSet, 2, 0, 1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gfxVisInfo, nullptr};
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
    VkShaderModule cullAtomicSM = loadShaderModule(shaderDir + "cull_atomic.comp.spv");
    VkShaderModule cullScanSM = loadShaderModule(shaderDir + "cull_scan.comp.spv");
    VkShaderModule vertSM = loadShaderModule(shaderDir + "cube.vert.spv");
    VkShaderModule fragSM = loadShaderModule(shaderDir + "cube.frag.spv");
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

    // 顶点输入 (仅立方体网格顶点，实例数据从 SSBO 读取)
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

    // 销毁 shader modules (已编译进 pipeline)
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

    float time = 0.0f;
    uint32_t frameCount = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        time += 0.016f; // ~60fps

        // ─── 相机绕原点旋转 ───
        float camRadius = 35.0f;
        float camAngle = time * 0.5f;
        glm::vec3 camPos(camRadius * cosf(camAngle), 15.0f, camRadius * sinf(camAngle));
        glm::vec3 camTarget(0.0f, 0.0f, 0.0f);
        glm::vec3 camUp(0.0f, 1.0f, 0.0f);

        glm::mat4 view = glm::lookAt(camPos, camTarget, camUp);
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                                          (float)swapExtent.width / (float)swapExtent.height, 0.1f, 100.0f);
        // Vulkan Y 轴翻转
        proj[1][1] *= -1.0f;

        glm::mat4 viewProj = proj * view;

        // ─── 更新 Cull UBO ───
        CullUBO cullUbo{};
        cullUbo.viewProj = viewProj;
        extractFrustumPlanes(viewProj, cullUbo.frustumPlanes);
        cullUbo.instanceCount = INSTANCE_COUNT;
        {
            void *data;
            vkMapMemory(device, cullUboMemory, 0, sizeof(CullUBO), 0, &data);
            memcpy(data, &cullUbo, sizeof(CullUBO));
            vkUnmapMemory(device, cullUboMemory);
        }

        // ─── 更新 Render UBO ───
        RenderUBO renderUbo{};
        renderUbo.view = view;
        renderUbo.proj = proj;
        {
            void *data;
            vkMapMemory(device, renderUboMemory, 0, sizeof(RenderUBO), 0, &data);
            memcpy(data, &renderUbo, sizeof(RenderUBO));
            vkUnmapMemory(device, renderUboMemory);
        }

        // ─── 重置 Indirect Buffer ───
        // 每帧开始前，将 instanceCount 设为 0，indexCount 设为 36
        {
            void *data;
            vkMapMemory(device, indirectMemory, 0, indirectBufSize, 0, &data);
            uint32_t *ptr = (uint32_t *)data;
            ptr[0] = CUBE_INDEX_COUNT; // indexCount = 36
            ptr[1] = 0;                // instanceCount = 0 (compute 填充)
            ptr[2] = 0;                // firstIndex = 0
            ptr[3] = 0;                // vertexOffset = 0
            ptr[4] = 0;                // firstInstance = 0
            vkUnmapMemory(device, indirectMemory);
        }

        // ─── 等待上一帧完成 ───
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailSem, nullptr, &imageIndex);

        // ─── 记录命令缓冲 ───
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginCI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginCI);

        // ═══ 阶段 1: Compute Dispatch (GPU Frustum Culling) ═══
        VkPipeline currentCullPipeline = (cullMode == 0) ? cullAtomicPipeline : cullScanPipeline;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, currentCullPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compPipelineLayout, 0, 1, &compDescSet, 0, nullptr);

        // 计算 dispatch 大小: ceil(instanceCount / localSize)
        uint32_t groupCount = (INSTANCE_COUNT + COMPUTE_LOCAL_SIZE - 1) / COMPUTE_LOCAL_SIZE;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        // ═══ Pipeline Barrier: Compute → Draw Indirect ═══
        // 作用:
        //   1. 确保 compute shader 对 indirect buffer 的写入完成
        //      (SHADER_WRITE → INDIRECT_COMMAND_READ)
        //   2. 确保 compute shader 对 visible ID buffer 的写入完成
        //      (SHADER_WRITE → SHADER_READ)
        //
        // 没有这个 barrier，GPU 可能在 compute 还没写完时就开始读取
        // indirect buffer 执行绘制，导致渲染结果错误或崩溃
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                                      // src: compute 完成后
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, // dst: indirect draw + vertex shader 开始前
                             0,
                             1, &memBarrier, // global memory barrier
                             0, nullptr,
                             0, nullptr);

        // ═══ 阶段 2: Graphics Render Pass ═══
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.05f, 0.05f, 0.1f, 1.0f}}; // 深蓝色背景
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

        // 绑定立方体网格
        VkBuffer vbs[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vbs, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // ═══ GPU Driven 核心: vkCmdDrawIndexedIndirect ═══
        // 绘制参数完全来自 GPU 写的 indirect buffer
        // CPU 不知道有多少实例可见，也不需要知道
        vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer, 0, 1, 0);

        vkCmdEndRenderPass(commandBuffer);
        vkEndCommandBuffer(commandBuffer);

        // ─── 提交命令 ───
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
            std::cout << "[帧 " << frameCount << "] 模式: "
                      << (cullMode == 0 ? "atomicAdd" : "Prefix Sum") << std::endl;
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
    destroyBuf(indirectBuffer, indirectMemory);
    destroyBuf(visibleBuffer, visibleMemory);
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
        {
            return i;
        }
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

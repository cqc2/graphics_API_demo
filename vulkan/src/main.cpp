#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

int main()
{
    std::cout << "=== Vulkan 初始化测试 ===" << std::endl;

    // 获取 Vulkan 实例扩展
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "可用的 Vulkan 扩展数量: " << extensionCount << std::endl;
    std::cout << "\n扩展列表:" << std::endl;
    for (const auto &ext : extensions)
    {
        std::cout << "  - " << ext.extensionName
                  << " (版本: " << ext.specVersion << ")" << std::endl;
    }

    // 创建 Vulkan 实例
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Demo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (result == VK_SUCCESS)
    {
        std::cout << "\n✓ Vulkan 实例创建成功!" << std::endl;

        // 获取物理设备
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        std::cout << "\n可用的物理设备数量: " << deviceCount << std::endl;

        if (deviceCount > 0)
        {
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            for (uint32_t i = 0; i < deviceCount; i++)
            {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[i], &props);

                std::cout << "\n设备 " << i << ":" << std::endl;
                std::cout << "  名称: " << props.deviceName << std::endl;
                std::cout << "  类型: ";
                switch (props.deviceType)
                {
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    std::cout << "独立 GPU";
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    std::cout << "集成 GPU";
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    std::cout << "虚拟 GPU";
                    break;
                default:
                    std::cout << "其他";
                }
                std::cout << std::endl;
                std::cout << "  Vulkan 版本: " << VK_VERSION_MAJOR(props.apiVersion)
                          << "." << VK_VERSION_MINOR(props.apiVersion) << std::endl;
            }
        }

        vkDestroyInstance(instance, nullptr);
    }
    else
    {
        std::cout << "\n✗ Vulkan 实例创建失败 (错误代码: " << result << ")" << std::endl;
    }

    return 0;
}

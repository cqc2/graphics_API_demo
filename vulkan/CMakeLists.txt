cmake_minimum_required(VERSION 3.10)
project(VulkanDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 Vulkan SDK 路径
set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
if(NOT VULKAN_SDK_PATH)
    set(VULKAN_SDK_PATH "/Users/mac/VulkanSDK/1.4.335.1/macOS")
endif()

message(STATUS "Using Vulkan SDK: ${VULKAN_SDK_PATH}")

# 查找 Vulkan
find_package(Vulkan REQUIRED)

# 包含目录
include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${VULKAN_SDK_PATH}/include
)

# 添加可执行文件
add_executable(vulkan_demo
    src/main.cpp
)

# 链接库
target_link_libraries(vulkan_demo
    ${Vulkan_LIBRARIES}
)

# macOS 特定设置
if(APPLE)
    # 链接 MoltenVK
    set(MOLTENVK_LIB "${VULKAN_SDK_PATH}/lib/libMoltenVK.dylib")
    if(EXISTS ${MOLTENVK_LIB})
        target_link_libraries(vulkan_demo ${MOLTENVK_LIB})
        message(STATUS "Linked MoltenVK: ${MOLTENVK_LIB}")
    endif()
    
    set_target_properties(vulkan_demo PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.example.vulkan-demo"
    )
endif()

message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")

cmake_minimum_required(VERSION 3.10)
project(VulkanDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 调试配置 - 生成调试符号用于 LLDB
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

# 设置 Vulkan SDK 路径
set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
if(NOT VULKAN_SDK_PATH)
    set(VULKAN_SDK_PATH "/Users/mac/VulkanSDK/1.4.335.1/macOS")
endif()

message(STATUS "Using Vulkan SDK: ${VULKAN_SDK_PATH}")

# 优先使用 Vulkan SDK 中的库
set(CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}" ${CMAKE_PREFIX_PATH})
link_directories("${VULKAN_SDK_PATH}/lib")

# 查找 Vulkan
find_package(Vulkan REQUIRED)

# 查找 GLM
find_package(glm REQUIRED)

# 查找 GLFW
find_package(glfw3 REQUIRED)

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
    glm::glm
    glfw
)

# macOS 特定设置
if(APPLE)    
    set_target_properties(vulkan_demo PROPERTIES
        BUILD_RPATH "${VULKAN_SDK_PATH}/lib"
        INSTALL_RPATH "${VULKAN_SDK_PATH}/lib"
        LINK_FLAGS "-Wl,-rpath,${VULKAN_SDK_PATH}/lib"
    )
endif()

message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")

cmake_minimum_required(VERSION 3.10)
project(MeshletDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Vulkan SDK 路径 ───
set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
if(NOT VULKAN_SDK_PATH)
    set(VULKAN_SDK_PATH "/Users/mac/VulkanSDK/1.4.335.1/macOS")
endif()
message(STATUS "Using Vulkan SDK: ${VULKAN_SDK_PATH}")

set(CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}" ${CMAKE_PREFIX_PATH})
link_directories("${VULKAN_SDK_PATH}/lib")

# ─── 查找依赖 ───
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

# ─── 查找 glslangValidator (SPIR-V 编译器) ───
find_program(GLSLANG_VALIDATOR glslangValidator
    HINTS "$ENV{VULKAN_SDK}/bin" "${VULKAN_SDK_PATH}/bin"
    REQUIRED)
message(STATUS "glslangValidator: ${GLSLANG_VALIDATOR}")

# ─── 编译 GLSL → SPIR-V ───
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_SOURCES
    ${SHADER_SRC_DIR}/meshlet_cull_atomic.comp
    ${SHADER_SRC_DIR}/meshlet_cull_scan.comp
    ${SHADER_SRC_DIR}/meshlet.vert
    ${SHADER_SRC_DIR}/meshlet.frag
)

set(SPIRV_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_FILE ${SHADER_OUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER} -o ${SPIRV_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling GLSL → SPIR-V: ${SHADER_NAME}"
    )
    list(APPEND SPIRV_FILES ${SPIRV_FILE})
endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_FILES})

# ─── 包含目录 ───
include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${VULKAN_SDK_PATH}/include
)

# ─── 可执行文件 ───
add_executable(meshlet_demo
    src/main.cpp
)
add_dependencies(meshlet_demo compile_shaders)

# ─── 将 SHADER_DIR 路径编译到可执行文件中 ───
target_compile_definitions(meshlet_demo PRIVATE
    SHADER_DIR="${SHADER_OUT_DIR}/"
)

# ─── 链接库 ───
target_link_libraries(meshlet_demo
    ${Vulkan_LIBRARIES}
    glm::glm
    glfw
)

# ─── macOS 设置 ───
if(APPLE)
    set_target_properties(meshlet_demo PROPERTIES
        BUILD_RPATH "${VULKAN_SDK_PATH}/lib"
        INSTALL_RPATH "${VULKAN_SDK_PATH}/lib"
        LINK_FLAGS "-Wl,-rpath,${VULKAN_SDK_PATH}/lib"
    )
endif()

message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")
message(STATUS "Shader SPIR-V output: ${SHADER_OUT_DIR}")

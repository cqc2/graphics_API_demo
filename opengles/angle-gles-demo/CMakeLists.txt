cmake_minimum_required(VERSION 3.15)
project(ANGLE_GLES_Demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# ANGLE 库配置
# ============================================================================
set(ANGLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Lib/angle-macos-arm64")

if(NOT EXISTS "${ANGLE_ROOT}/libEGL.dylib")
    message(FATAL_ERROR "ANGLE libraries not found at ${ANGLE_ROOT}")
endif()

message(STATUS "ANGLE Root: ${ANGLE_ROOT}")

# ============================================================================
# 依赖：SDL2（窗口管理）
# ============================================================================
find_package(SDL2 REQUIRED)

# ============================================================================
# 依赖：GLM（数学库 - 仅头文件）
# ============================================================================
# GLM 安装在 Homebrew 的 ARM64 路径
set(GLM_INCLUDE_DIR "/opt/homebrew/include")

# ============================================================================
# 源文件
# ============================================================================
set(SOURCES
    src/main.cpp
)

# ============================================================================
# 创建可执行文件
# ============================================================================
add_executable(angle_gles_demo ${SOURCES})

# ============================================================================
# 包含目录
# ============================================================================
target_include_directories(angle_gles_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ANGLE_ROOT}/include
    ${SDL2_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIR}
)

# ============================================================================
# 链接库 - 使用 -rpath 确保从可执行文件目录加载
# ============================================================================
target_link_libraries(angle_gles_demo PRIVATE
    -L${ANGLE_ROOT}
    -lEGL
    -lGLESv2
    ${SDL2_LIBRARIES}
    "-framework CoreFoundation"
    "-Wl,-rpath,@executable_path"
)

# ============================================================================
# 设置 macOS 特定属性
# ============================================================================
set_target_properties(angle_gles_demo PROPERTIES
    MACOSX_RPATH ON
)

# ============================================================================
# 复制库文件到构建目录并修复链接
# ============================================================================
add_custom_command(TARGET angle_gles_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${ANGLE_ROOT}/libEGL.dylib"
        "$<TARGET_FILE_DIR:angle_gles_demo>/libEGL.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${ANGLE_ROOT}/libGLESv2.dylib"
        "$<TARGET_FILE_DIR:angle_gles_demo>/libGLESv2.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy
        "/opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib"
        "$<TARGET_FILE_DIR:angle_gles_demo>/libSDL2-2.0.0.dylib"
    # 修复库的安装名称为可执行文件相对路径
    COMMAND install_name_tool -change ./libEGL.dylib @executable_path/libEGL.dylib
        "$<TARGET_FILE:angle_gles_demo>"
    COMMAND install_name_tool -change ./libGLESv2.dylib @executable_path/libGLESv2.dylib
        "$<TARGET_FILE:angle_gles_demo>"
    COMMAND install_name_tool -change /opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib @executable_path/libSDL2-2.0.0.dylib
        "$<TARGET_FILE:angle_gles_demo>"
    COMMENT "Copying ANGLE libraries and fixing library paths"
)

# ============================================================================
# 编译选项
# ============================================================================
if(APPLE)
    set_target_properties(angle_gles_demo PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.example.angle-gles-demo"
        MACOSX_BUNDLE_BUNDLE_NAME "ANGLE GLES Demo"
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SDL2 found: ${SDL2_FOUND}")

cmake_minimum_required(VERSION 3.20)
project(SlangKernelLauncher VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── SlangCompiler 子模块 ──────────────────────────────────────────────────────
# 构建后提供：
#   SlangCompiler          — 编译器 exe
#   SlangKernelHeaders     — custom target，负责跑 exe 生成所有 .h
#   SLANG_GENERATED_DIR    — 生成头文件所在目录（CACHE INTERNAL）
add_subdirectory(compiler)

# ── CUDA Driver API 检测（与 compiler/ 子模块独立，只需要 driver 库）────────────
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found (${CUDAToolkit_VERSION}): CUDA Driver backend enabled")
else()
    message(STATUS "CUDAToolkit not found: CUDA Driver backend disabled")
endif()

# ── SlangKernelLauncher 库 ────────────────────────────────────────────────────
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.h"
)

add_library(${PROJECT_NAME} STATIC
    ${LIB_SOURCES}
    ${SLANG_GENERATED_CPU_SOURCES}   # Slang 生成的 CPU C++ 源文件，用于 CPU 端调试
)

# 依赖 SlangKernelHeaders target，确保头文件在编译本库之前生成完毕
add_dependencies(${PROJECT_NAME} SlangKernelHeaders)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${SLANG_GENERATED_DIR}   # compiler/ 子模块暴露的生成头文件目录
)

# CUDA Driver backend
if(CUDAToolkit_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC SKL_HAS_CUDA_DRIVER)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
endif()

# ── 平台特定设置 ──────────────────────────────────────────────────────────────
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(MACOS_SDK_PATH)
        target_compile_options(${PROJECT_NAME} PUBLIC
            -isystem "${MACOS_SDK_PATH}/usr/include/c++/v1"
        )
    endif()
endif()

# ── 测试 ──────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_kernel_launch tests/test_kernel_launch.cpp)

add_dependencies(test_kernel_launch SlangKernelHeaders)

target_link_libraries(test_kernel_launch PRIVATE ${PROJECT_NAME})

target_include_directories(test_kernel_launch PRIVATE
    ${SLANG_GENERATED_DIR}   # 生成的 kernel.h
)

add_test(NAME KernelLaunchTest COMMAND test_kernel_launch)

# ── CPU kernel 正确性测试 ──────────────────────────────────────────────────────
add_executable(test_cpu_kernel tests/test_cpu_kernel.cpp)

add_dependencies(test_cpu_kernel SlangKernelHeaders)

# 链接库（内含 kernel_cpu.cpp），PUBLIC include 自动带入 include/ 目录
target_link_libraries(test_cpu_kernel PRIVATE ${PROJECT_NAME})

target_include_directories(test_cpu_kernel PRIVATE
    ${SLANG_GENERATED_DIR}   # slang_kernels.h / kernel_cpu.h
)

add_test(NAME CpuKernelTest COMMAND test_cpu_kernel)

# ── KernelFactory 测试 ─────────────────────────────────────────────────────────
add_executable(test_kernel_factory tests/test_kernel_factory.cpp)

add_dependencies(test_kernel_factory SlangKernelHeaders)

target_link_libraries(test_kernel_factory PRIVATE ${PROJECT_NAME})

target_include_directories(test_kernel_factory PRIVATE
    ${SLANG_GENERATED_DIR}   # slang_kernels.h / kernel_factory.h
)

add_test(NAME KernelFactoryTest COMMAND test_kernel_factory)

# ── 安装规则 ──────────────────────────────────────────────────────────────────
install(TARGETS ${PROJECT_NAME}
    EXPORT SlangKernelLauncherTargets
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT SlangKernelLauncherTargets
    NAMESPACE SlangKernelLauncher::
    DESTINATION lib/cmake/SlangKernelLauncher
)

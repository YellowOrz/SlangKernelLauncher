cmake_minimum_required(VERSION 3.20)
project(SlangCompiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ── 查找 / 自动下载 Slang SDK ─────────────────────────────────────────────────
set(SLANG_VERSION   "2026.3.1")
set(SLANG_SDK_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/slang")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(SLANG_CMAKE_DIR   "${SLANG_SDK_DIR}/cmake")
else()
    set(SLANG_CMAKE_DIR   "${SLANG_SDK_DIR}/lib/cmake/slang")
endif()
set(_slang_config   "${SLANG_CMAKE_DIR}/slangConfig.cmake")

if(NOT EXISTS "${_slang_config}")
    message(STATUS "Slang SDK not found at ${SLANG_SDK_DIR} — downloading v${SLANG_VERSION}...")

    # ── 平台/架构检测 ──────────────────────────────────────────────────────────
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(_slang_platform "macos-aarch64")
        else()
            set(_slang_platform "macos-x86_64")
        endif()
        set(_slang_ext "zip")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
            set(_slang_platform "windows-aarch64")
        else()
            set(_slang_platform "windows-x86_64")
        endif()
        set(_slang_ext "zip")
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(_slang_platform "linux-aarch64")
        else()
            set(_slang_platform "linux-x86_64")
        endif()
        set(_slang_ext "tar.gz")
    endif()

    set(_slang_url
        "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-${_slang_platform}.${_slang_ext}"
    )
    set(_slang_archive
        "${CMAKE_CURRENT_BINARY_DIR}/slang-${SLANG_VERSION}-${_slang_platform}.${_slang_ext}"
    )

    message(STATUS "  URL: ${_slang_url}")
    file(DOWNLOAD "${_slang_url}" "${_slang_archive}"
        SHOW_PROGRESS
        STATUS  _dl_status
        TLS_VERIFY ON
    )
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        list(GET _dl_status 1 _dl_msg)
        message(FATAL_ERROR "Failed to download Slang SDK: ${_dl_msg}\n  URL: ${_slang_url}")
    endif()

    file(MAKE_DIRECTORY "${SLANG_SDK_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${_slang_archive}" DESTINATION "${SLANG_SDK_DIR}")
    file(REMOVE "${_slang_archive}")
    message(STATUS "Slang SDK v${SLANG_VERSION} installed at ${SLANG_SDK_DIR}")
endif()

find_package(slang CONFIG
    PATHS "${SLANG_CMAKE_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET Slang::Slang)
    add_library(Slang::Slang ALIAS slang::slang)
endif()
message(STATUS "Found Slang v${SLANG_VERSION} at ${SLANG_SDK_DIR}")

# ── 环境检测：Vulkan / Metal / CUDA ──────────────────────────────────────────

# Vulkan：有 SDK 才启用 SPIR-V target（否则生成的字节码无处可用）
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK found: vulkan target enabled")
else()
    message(STATUS "Vulkan SDK not found: vulkan target disabled")
endif()

# Metal：仅 Apple 平台
if(APPLE)
    message(STATUS "Apple platform: metal target enabled")
endif()

# CUDA：通过 CUDAToolkit 检测，获取 NVRTC 库
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found (${CUDAToolkit_VERSION}): cuda target enabled (NVRTC)")
else()
    message(STATUS "CUDA Toolkit not found: cuda target disabled")
endif()

# ── SlangCompiler 可执行文件 ──────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")

add_executable(SlangCompiler ${SOURCES})
target_link_libraries(SlangCompiler PRIVATE Slang::Slang)

# 把检测结果以宏定义传入源码
if(Vulkan_FOUND)
    target_compile_definitions(SlangCompiler PRIVATE SLANG_COMPILER_HAS_VULKAN)
endif()
if(APPLE)
    target_compile_definitions(SlangCompiler PRIVATE SLANG_COMPILER_HAS_METAL)
endif()
if(CUDAToolkit_FOUND)
    target_compile_definitions(SlangCompiler PRIVATE SLANG_COMPILER_HAS_CUDA)
    target_link_libraries(SlangCompiler PRIVATE CUDA::nvrtc)
    # NVRTC 运行时库也需要在运行时找到
    target_link_libraries(SlangCompiler PRIVATE CUDA::cudart)
endif()

# ── 平台特定设置 ──────────────────────────────────────────────────────────────
if(WIN32)
    add_custom_command(TARGET SlangCompiler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:SlangCompiler>
            $<TARGET_FILE_DIR:SlangCompiler>
        COMMAND_EXPAND_LISTS
    )
elseif(APPLE)
    set_target_properties(SlangCompiler PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(MACOS_SDK_PATH)
        target_compile_options(SlangCompiler PRIVATE
            -isystem "${MACOS_SDK_PATH}/usr/include/c++/v1"
        )
    endif()
elseif(UNIX)
    set_target_properties(SlangCompiler PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

if(slang_FOUND AND NOT WIN32)
    add_custom_command(TARGET SlangCompiler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:slang::slang>
            $<TARGET_FILE_DIR:SlangCompiler>
        COMMENT "Copying slang runtime..."
    )
endif()

# ── SlangKernelHeaders target ─────────────────────────────────────────────────
# 收集 shaders/ 下所有 .slang 文件
file(GLOB_RECURSE SLANG_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang"
)

# 生成头文件统一放在 build/generated/ 下
set(SLANG_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
    CACHE INTERNAL "Directory containing SlangCompiler-generated headers")
file(MAKE_DIRECTORY "${SLANG_GENERATED_DIR}")

# cmake --build . --target clean 时整体删除生成目录，消除孤立文件
set_directory_properties(PROPERTIES
    ADDITIONAL_CLEAN_FILES "${SLANG_GENERATED_DIR}"
)

set(_generated_headers)
set(_generated_cpu_sources)

foreach(_slang_src ${SLANG_SOURCES})
    get_filename_component(_name "${_slang_src}" NAME_WE)
    set(_out_h          "${SLANG_GENERATED_DIR}/${_name}.h")
    set(_out_cpu        "${SLANG_GENERATED_DIR}/${_name}_cpu.cpp")
    set(_out_cpu_h      "${SLANG_GENERATED_DIR}/${_name}_cpu.h")
    set(_out_factory_h  "${SLANG_GENERATED_DIR}/${_name}_factory.h")

    add_custom_command(
        OUTPUT  "${_out_h}" "${_out_cpu}" "${_out_cpu_h}" "${_out_factory_h}"
        COMMAND SlangCompiler "${_slang_src}" "${_out_h}" "${_name}"
        DEPENDS SlangCompiler "${_slang_src}"
        COMMENT "Slang: ${_name}.slang -> ${_name}.h + ${_name}_cpu.{h,cpp} + ${_name}_factory.h (entry: ${_name})"
    )
    list(APPEND _generated_headers     "${_out_h}")
    list(APPEND _generated_cpu_sources "${_out_cpu}")
endforeach()

# ── slang_kernels.h：聚合所有生成头文件的 include helper ─────────────────────
# 在 CMake 配置期生成（.slang 文件列表在 configure 时已知，增删 .slang 须重跑 cmake）
set(SLANG_HELPER_HEADER "${SLANG_GENERATED_DIR}/slang_kernels.h"
    CACHE INTERNAL "Aggregator header that includes all Slang-generated kernel headers")

file(WRITE "${SLANG_HELPER_HEADER}"
    "#pragma once\n"
    "// Auto-generated by CMake — DO NOT EDIT\n"
    "// Include all Slang-generated kernel headers\n"
    "\n"
)

# 收集 kernel 名称列表，用于生成枚举和名称表
set(_kernel_names)
foreach(_slang_src ${SLANG_SOURCES})
    get_filename_component(_name "${_slang_src}" NAME_WE)
    list(APPEND _kernel_names "${_name}")
    file(APPEND "${SLANG_HELPER_HEADER}" "#include \"${_name}.h\"\n")
    file(APPEND "${SLANG_HELPER_HEADER}" "#include \"${_name}_cpu.h\"\n")
endforeach()

# 枚举值：每个 kernel 对应一个成员，末尾加 Count
file(APPEND "${SLANG_HELPER_HEADER}" "\nenum class SlangKernelID {\n")
set(_idx 0)
foreach(_name ${_kernel_names})
    file(APPEND "${SLANG_HELPER_HEADER}" "    ${_name} = ${_idx},\n")
    math(EXPR _idx "${_idx} + 1")
endforeach()
file(APPEND "${SLANG_HELPER_HEADER}" "    Count = ${_idx}\n")
file(APPEND "${SLANG_HELPER_HEADER}" "};\n\n")

# 名称字符串表（与枚举下标一一对应）
file(APPEND "${SLANG_HELPER_HEADER}"
    "#ifndef SLANG_KERNEL_NAMES_IMPL\n"
    "extern const char* const slang_kernel_names[];\n"
    "#else\n"
    "const char* const slang_kernel_names[] = {\n"
)
foreach(_name ${_kernel_names})
    file(APPEND "${SLANG_HELPER_HEADER}" "    \"${_name}\",\n")
endforeach()
file(APPEND "${SLANG_HELPER_HEADER}"
    "};\n"
    "#endif // SLANG_KERNEL_NAMES_IMPL\n"
)

# ── KernelFactory 特化 ────────────────────────────────────────────────────────
# 特化代码由 SlangCompiler 在构建时生成到各自的 kernel_factory.h 中。
# 此处只负责在 SlangKernelID 枚举定义后 include 这些文件。
file(APPEND "${SLANG_HELPER_HEADER}" "\n// KernelFactory specializations (generated by SlangCompiler)\n")
foreach(_name ${_kernel_names})
    file(APPEND "${SLANG_HELPER_HEADER}" "#include \"${_name}_factory.h\"\n")
endforeach()

message(STATUS "Generated ${SLANG_HELPER_HEADER}")

# 上层使用：
#   add_dependencies(MyLib SlangKernelHeaders)
#   target_sources(MyLib PRIVATE ${SLANG_GENERATED_CPU_SOURCES})
#   target_include_directories(MyLib PRIVATE ${SLANG_GENERATED_DIR})
#   #include "slang_kernels.h"  ← 一行包含所有 kernel
add_custom_target(SlangKernelHeaders
    DEPENDS ${_generated_headers} ${_generated_cpu_sources}
    COMMENT "Compiling all Slang kernels..."
)

set(SLANG_GENERATED_CPU_SOURCES "${_generated_cpu_sources}"
    CACHE INTERNAL "Generated CPU C++ source files from Slang kernels")

# ── rebuild-shaders target ────────────────────────────────────────────────────
# 删除 .slang 文件后使用此 target 清除孤立的生成文件并重新全量生成：
#   cmake --build . --target rebuild-shaders
add_custom_target(rebuild-shaders
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${SLANG_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SLANG_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target SlangKernelHeaders
    COMMENT "Cleaning and regenerating all Slang kernel headers..."
)

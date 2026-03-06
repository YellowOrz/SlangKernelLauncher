# FindSlang.cmake
# ────────────────────────────────────────────────────────────────────────────
# 在以下位置查找 Slang SDK（按优先级排序）：
#   1. 环境变量   SLANG_DIR / SLANG_SDK_DIR
#   2. CMake 变量 Slang_DIR / SLANG_DIR（cmake -DSlang_DIR=...）
#   3. 常见系统默认安装路径（各平台）
#
# 找到后导出：
#   Slang::Slang          — 可链接目标
#   Slang_FOUND           — 是否找到
#   Slang_INCLUDE_DIR     — 头文件目录
#   Slang_LIBRARY         — 库路径
#   Slang_DLL             — (Windows) DLL 路径
# ────────────────────────────────────────────────────────────────────────────

# 候选搜索根目录
set(_slang_search_roots
    "$ENV{SLANG_DIR}"
    "$ENV{SLANG_SDK_DIR}"
    "${Slang_DIR}"
    "${SLANG_DIR}"
)

if(WIN32)
    list(APPEND _slang_search_roots
        "C:/slang"
        "C:/Program Files/slang"
        "$ENV{ProgramFiles}/slang"
    )
elseif(APPLE)
    list(APPEND _slang_search_roots
        "/usr/local"
        "/opt/homebrew"
        "/opt/slang"
        "$ENV{HOME}/slang"
    )
elseif(UNIX)
    list(APPEND _slang_search_roots
        "/usr"
        "/usr/local"
        "/opt/slang"
        "$ENV{HOME}/slang"
    )
endif()

# 移除空项，避免无效路径干扰
list(REMOVE_ITEM _slang_search_roots "")

# ── 查找头文件 ──────────────────────────────────────────────────────────────
find_path(Slang_INCLUDE_DIR
    NAMES slang.h
    PATHS ${_slang_search_roots}
    PATH_SUFFIXES include
    DOC "Slang include directory"
)

# ── 查找库文件 ──────────────────────────────────────────────────────────────
if(WIN32)
    find_library(Slang_LIBRARY
        NAMES slang
        PATHS ${_slang_search_roots}
        PATH_SUFFIXES lib lib/x64 lib/x86_64 bin
        DOC "Slang import library (Windows)"
    )
    # 同时找 DLL，供运行时拷贝
    find_file(Slang_DLL
        NAMES slang.dll
        PATHS ${_slang_search_roots}
        PATH_SUFFIXES bin bin/x64 bin/x86_64 lib
        DOC "Slang DLL (Windows)"
    )
elseif(APPLE)
    find_library(Slang_LIBRARY
        NAMES slang
        PATHS ${_slang_search_roots}
        PATH_SUFFIXES lib lib/x64 lib/x86_64
        DOC "Slang shared library (macOS)"
    )
elseif(UNIX)
    find_library(Slang_LIBRARY
        NAMES slang
        PATHS ${_slang_search_roots}
        PATH_SUFFIXES lib lib/x64 lib/x86_64
        DOC "Slang shared library (Linux)"
    )
endif()

# ── 标准处理（设置 Slang_FOUND，输出错误信息）───────────────────────────────
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Slang
    REQUIRED_VARS Slang_LIBRARY Slang_INCLUDE_DIR
    FAIL_MESSAGE
        "Cannot find Slang SDK. Set SLANG_DIR environment variable or pass "
        "-DSlang_DIR=<path> to cmake. "
        "Download from: https://github.com/shader-slang/slang/releases"
)

# ── 创建导入目标 Slang::Slang ────────────────────────────────────────────────
if(Slang_FOUND AND NOT TARGET Slang::Slang)
    add_library(Slang::Slang SHARED IMPORTED)

    set_target_properties(Slang::Slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Slang_INCLUDE_DIR}"
    )

    if(WIN32)
        set_target_properties(Slang::Slang PROPERTIES
            IMPORTED_IMPLIB "${Slang_LIBRARY}"
            IMPORTED_LOCATION "${Slang_DLL}"
        )
    else()
        set_target_properties(Slang::Slang PROPERTIES
            IMPORTED_LOCATION "${Slang_LIBRARY}"
        )
    endif()
endif()

mark_as_advanced(Slang_INCLUDE_DIR Slang_LIBRARY Slang_DLL)

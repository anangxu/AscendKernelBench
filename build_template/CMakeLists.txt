# AscendKernelBench kernel build template: compiles custom_op.asc into a
# process-local libcustom_op.so loaded with torch.ops.load_library. Never an
# OPP package (no custom_opp_*.run), never a pybind module.

# With -DAKB_SPLIT_BUILD=ON the sample directory instead provides
# custom_op_kernel.asc (device code), custom_op_launch.asc (generated launch
# stubs), and custom_op_host.cpp (torch glue).

# The host glue compiles as plain C++ against the shared precompiled header
# AKB_PCH_PATH, so the torch headers are parsed once per environment instead
# of once per sample.

# CMAKE_ASC_ARCHITECTURES is injected by build.py from the hardware profile.
# Requires CANN 9.1.0 or newer, with torch and torch_npu importable.

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_ASC_ARCHITECTURES OR "${CMAKE_ASC_ARCHITECTURES}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_ASC_ARCHITECTURES is required (e.g. dav-2201, dav-3510)")
endif()
set(CMAKE_ASC_ARCHITECTURES "${CMAKE_ASC_ARCHITECTURES}" CACHE STRING "NPU architecture" FORCE)

# Keeps cmake --install out of site-packages and ASCEND_OPP_PATH/vendors.
set(CMAKE_SKIP_INSTALL_RULES ON)

find_package(ASC REQUIRED)
project(custom_op_build LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# gnu++17 vs c++17 changes predefined macros, which invalidates the PCH.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AKB_SPLIT_BUILD "Build from split device/host sources" OFF)
set(AKB_PCH_PATH "" CACHE FILEPATH "Shared precompiled header for the host TU")

# Optional ccache, enabled with -DENABLE_CCACHE=ON. The ASC launcher matters
# most: the .asc translation unit dominates build time.
option(ENABLE_CCACHE "Enable ccache compiler launcher" OFF)
if(ENABLE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "[ccache] Enabled: ${CCACHE_PROGRAM}")
        set(CMAKE_ASC_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else()
        message(WARNING "[ccache] ENABLE_CCACHE=ON but ccache was not found")
    endif()
endif()

# The ASC/bisheng front-end is LLVM based, so it needs an explicit
# --gcc-toolchain to resolve C++ standard library headers. Query plain gcc:
# in split mode CMAKE_CXX_COMPILER is bisheng itself.
find_program(AKB_GCC_EXECUTABLE gcc REQUIRED)
execute_process(
    COMMAND ${AKB_GCC_EXECUTABLE} -print-file-name=libgcc.a
    OUTPUT_VARIABLE LIBGCC_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GCC_QUERY_RESULT
)
if(NOT GCC_QUERY_RESULT EQUAL 0 OR NOT EXISTS "${LIBGCC_PATH}")
    message(FATAL_ERROR "Failed to locate libgcc.a using ${AKB_GCC_EXECUTABLE}")
endif()
get_filename_component(LIBGCC_PATH "${LIBGCC_PATH}" REALPATH)
string(REGEX REPLACE "/lib/gcc/.*$" "" GCC_TOOLCHAIN_ROOT "${LIBGCC_PATH}")
if("${GCC_TOOLCHAIN_ROOT}" STREQUAL "${LIBGCC_PATH}")
    message(FATAL_ERROR "Failed to derive GCC toolchain root from ${LIBGCC_PATH}")
endif()
message(STATUS "GCC toolchain root: ${GCC_TOOLCHAIN_ROOT}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# build.py passes TORCH_CMAKE_PREFIX_PATH and TORCH_NPU_PATH from the
# worker's own torch import; the template itself does not discover Python.
if(NOT DEFINED TORCH_CMAKE_PREFIX_PATH OR NOT DEFINED TORCH_NPU_PATH)
    message(FATAL_ERROR "TORCH_CMAKE_PREFIX_PATH and TORCH_NPU_PATH are required (supplied by build.py)")
endif()
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

if(AKB_SPLIT_BUILD)
    # Split mode: device kernels, generated launch stubs, and torch glue are
    # separate translation units linked into one process-local library.
    add_library(custom_op SHARED
        custom_op_kernel.asc
        custom_op_launch.asc
        custom_op_host.cpp
    )
    # The host TU is thin glue, so -O1 trades no measurable runtime. The
    # ASC module empties CMAKE_CXX_FLAGS_RELEASE, hence the explicit flags;
    # they must match HOST_TU_FLAGS in build.py for PCH validity.
    set(_AKB_HOST_TU_OPTIONS
        -O1
        -DNDEBUG
        --gcc-toolchain=${GCC_TOOLCHAIN_ROOT}
        -include ${CMAKE_CURRENT_SOURCE_DIR}/custom_op_launch_decls.h
    )
    if(NOT "${AKB_PCH_PATH}" STREQUAL "")
        list(APPEND _AKB_HOST_TU_OPTIONS -include-pch ${AKB_PCH_PATH})
    endif()
    set_source_files_properties(custom_op_host.cpp PROPERTIES
        COMPILE_OPTIONS "${_AKB_HOST_TU_OPTIONS}"
    )
else()
    # Shared library loaded with torch.ops.load_library (not a pybind module).
    add_library(custom_op SHARED custom_op.asc)
endif()

set_target_properties(custom_op PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "custom_op"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(custom_op PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Python3::Python is deliberately not linked: the library is dlopen'd, never
# imported, and references no Python C API symbols. A NEEDED entry on
# libpython breaks loading under statically linked interpreters.
target_link_libraries(custom_op PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
)

target_link_directories(custom_op PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

########################## RPATH ##########################
# The library is loaded from the sample directory, so it must find Torch,
# torch_npu and CANN without a global install step.
set(_CUSTOM_OP_RPATH "$ORIGIN")
if(DEFINED TORCH_INSTALL_PREFIX AND NOT "${TORCH_INSTALL_PREFIX}" STREQUAL "")
    list(APPEND _CUSTOM_OP_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()
if(EXISTS "${TORCH_NPU_PATH}/lib")
    list(APPEND _CUSTOM_OP_RPATH "${TORCH_NPU_PATH}/lib")
endif()
if(DEFINED ENV{ASCEND_HOME_PATH})
    if(EXISTS "$ENV{ASCEND_HOME_PATH}/lib64")
        list(APPEND _CUSTOM_OP_RPATH "$ENV{ASCEND_HOME_PATH}/lib64")
    endif()
    if(EXISTS "$ENV{ASCEND_HOME_PATH}/lib")
        list(APPEND _CUSTOM_OP_RPATH "$ENV{ASCEND_HOME_PATH}/lib")
    endif()
endif()
set_property(TARGET custom_op PROPERTY BUILD_RPATH ${_CUSTOM_OP_RPATH})
message(STATUS "custom_op RPATH: ${_CUSTOM_OP_RPATH}")
########################## RPATH ##########################

target_link_options(custom_op PRIVATE
    --gcc-toolchain=${GCC_TOOLCHAIN_ROOT}
)

target_compile_options(custom_op PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--gcc-toolchain=${GCC_TOOLCHAIN_ROOT}>
)

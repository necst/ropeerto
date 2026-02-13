cmake_minimum_required(VERSION 3.21)
set(CMAKE_C_COMPILER   "/opt/rocm/bin/hipcc" CACHE PATH "C compiler")
set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE PATH "C++ compiler")
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/mutual_information.cpp" CACHE PATH "Path to the source files")
#set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type")
project(p2p_baseline LANGUAGES CXX HIP)


option(COYOTE_SUPPORT "Enable Coyote support" ON)
option(HW_REG "Enable MI computation on hardware" ON)

if(COYOTE_SUPPORT)
    message(STATUS "Coyote support enabled.")
    add_compile_definitions(COYOTE_MODE)
else()
    message(STATUS "Coyote support disabled. XRT used.")
endif()

if (HW_REG)
    message(STATUS "MI computation on hardware.")
    add_compile_definitions(HW_REG)
else()
    message(STATUS "MI computation in software.")
endif()


# --------------------------------------------------------
# 1) Impostazioni ROCm/HIP
# --------------------------------------------------------
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
    endif()

    file(STRINGS "${ROCM_PATH}/.info/version" ROCM_VERSION)
    message("-- Found ROCm: ${ROCM_VERSION}")

    if (NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
    endif()

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()

    if(NOT DEFINED HCC_PATH)
        if(DEFINED ENV{HCC_PATH})
            set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
        else()
            set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
        endif()
        set(HCC_HOME "${HCC_PATH}")
    endif()

    if(NOT DEFINED HIP_CLANG_PATH)
        if(NOT DEFINED ENV{HIP_CLANG_PATH})
            set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
        else()
            set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
        endif()
    endif()

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake" )
    list(APPEND CMAKE_PREFIX_PATH
        "${HIP_PATH}/lib/cmake"
        "${HIP_PATH}/../lib/cmake"
    )

    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()
    find_package(hip REQUIRED)

    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908"
    CACHE STRING "HIP GPU architectures to compile for")
    
# --------------------------------------------------------
# 3) Sorgenti e include
# --------------------------------------------------------
set(SOURCES
  ${SRC}
  HIPRigidWarp3D/src/utils/images_io.cpp
  HIPRigidWarp3D/src/utils/hip_host_helpers.cpp
  HIPRigidWarp3D/src/utils/args_parser.cpp
  HIPRigidWarp3D/src/utils/timer.cpp
  HIPRigidWarp3D/src/hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hip
  irg_app/HAL/HardwareAbstractionLayer.cpp
)

add_executable(p2p_baseline ${SOURCES})

if(COYOTE_SUPPORT)

# --------------------------------------------------------
# 2) Coyote (host runtime)
# --------------------------------------------------------
# Coyote (in-tree build via add_subdirectory)
set(CYT_DIR "${CMAKE_SOURCE_DIR}/../../../Coyote")

# Out-of-tree add_subdirectory requires an explicit binary dir
add_subdirectory("${CYT_DIR}/sw" "${CMAKE_BINARY_DIR}/coyote-sw")
target_link_libraries(p2p_baseline PRIVATE Coyote)
target_include_directories(p2p_baseline PRIVATE ${CYT_DIR}/sw/include/coyote)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)

target_link_libraries(p2p_baseline PUBLIC Coyote)

else()

# --------------------------------------------------------
# 2) XRT (host runtime)
# --------------------------------------------------------
# XRT

if (NOT EXISTS $ENV{XILINX_XRT})
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()

target_link_directories(p2p_baseline PUBLIC $ENV{XILINX_XRT}/lib)
target_link_libraries(p2p_baseline PUBLIC xilinxopencl xrt_coreutil xrt_core   ${XRT_COREUTIL_LIB})
target_include_directories(p2p_baseline PUBLIC $ENV{XILINX_XRT}/include)

if (NOT EXISTS $ENV{XILINX_HLS})
  message(WARNING "Xilinx HLS not loaded, falling back to open-source HLS headers")
  set(HLS_INCLUDES ${HLS_HEADERS_DIR} ${HLSLIB_INCLUDE})
  if (NOT EXISTS ${HLS_HEADERS_DIR})
    message(FATAL_ERROR "Open-source HLS headers submodule not loaded! Run git submodule update --init --recursive in ${ACCL_REPO_ROOT}")
  endif (NOT EXISTS ${HLS_HEADERS_DIR})
else ()
  message(NOTICE "Using Xilinx HLS headers")
  set(HLS_INCLUDES $ENV{XILINX_HLS}/include/ ${HLSLIB_INCLUDE})
endif (NOT EXISTS $ENV{XILINX_HLS})

target_include_directories(p2p_baseline PRIVATE
  ${XRT_INCLUDE_DIR}
  ${HLS_INCLUDES}
  ${HLSLIB_INCLUDE}
)

endif()

target_include_directories(p2p_baseline PRIVATE
  $<TARGET_PROPERTY:hip::device,INTERFACE_INCLUDE_DIRECTORIES>
  HIPRigidWarp3D/src/utils
  HIPRigidWarp3D/src
  HIPRigidWarp3D/src/hip_kernels
  HIPRigidWarp3D/externals
  irg_app/core
  irg_app/app
  irg_app/include
  irg_app/infrastructure
  irg_app/interfaces
  ${CMAKE_SOURCE_DIR}/../hw/src/hls/mutual_information_master
  $<BUILD_INTERFACE:${ROCM_PATH}/include>
  $<BUILD_INTERFACE:${ROCM_PATH}/include/hsa>
)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/../hw/src/hls/mutual_information_master/constants.h")
  message(FATAL_ERROR "constants.h not found. Build/configure hw first to generate it.")
endif()

# --------------------------------------------------------
# 3) OpenCV
# --------------------------------------------------------
# Use OPENCV_DIR if provided via environment.
# It must point to a directory containing OpenCVConfig.cmake
# (typically: <opencv_build>/build).

if(DEFINED ENV{OPENCV_DIR} AND NOT "$ENV{OPENCV_DIR}" STREQUAL "")
  set(OpenCV_DIR "$ENV{OPENCV_DIR}")
else()
  message(FATAL_ERROR
    "OPENCV_DIR is not set.\n"
    "Please export it as:\n"
    "  export OPENCV_DIR=<path to build folder in opencv_build>\n"
    "Example:\n"
    "  export OPENCV_DIR=$HOME/opencv_build/opencv/build"
  )
endif()

find_package(OpenCV REQUIRED)
target_include_directories(p2p_baseline PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(p2p_baseline PRIVATE ${OpenCV_LIBS})

# --------------------------------------------------------
# 4) Link libraries
# --------------------------------------------------------
target_link_libraries(p2p_baseline PRIVATE
  pthread
  hsa-runtime64
  hip::device
  hsakmt
)

# --------------------------------------------------------
# 5) Opzioni di compilazione
# --------------------------------------------------------
target_compile_features(p2p_baseline PRIVATE cxx_std_17)
target_compile_options(p2p_baseline PRIVATE -O3)

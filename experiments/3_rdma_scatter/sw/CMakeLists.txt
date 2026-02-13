######################################################################################
# This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
# 
# MIT Licence
# Copyright (c) 2025, Systems Group, ETH Zurich
# All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
######################################################################################

# CMake configuration
cmake_minimum_required(VERSION 3.5)
project(rdma_scatter_artifact_sw)

# Add Coyote sources
add_subdirectory(../../../Coyote/sw ${CMAKE_BINARY_DIR}/coyote)

# Add source files
set(INSTANCE "client_p2p" CACHE STRING "RDMA software build target: client_p2p, client_non_p2p or server")
if(INSTANCE STREQUAL "server")
    set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src/server")
    message("*** RDMA Server [Software] ***")
    include_directories("${CMAKE_SOURCE_DIR}/src/include")
endif()
if(INSTANCE STREQUAL "client_p2p")
    set(AMDGPU_TARGETS "gfx90a" CACHE STRING "Target GPU platform")
    set(EN_GPU ON)
    set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src/client_p2p")
    message("*** RDMA Client P2P [Software] ***")
    include_directories("${CMAKE_SOURCE_DIR}/src/include")
endif()
if(INSTANCE STREQUAL "client_non_p2p")
    set(AMDGPU_TARGETS "gfx90a" CACHE STRING "Target GPU platform")
    set(EN_GPU ON)
    set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src/client_non_p2p")
    message("*** RDMA Client Non-P2P [Software] ***")
    include_directories("${CMAKE_SOURCE_DIR}/src/include")
endif()

# Create build targets and link against required libraries
set(EXEC test)
add_executable(${EXEC} ${TARGET_DIR}/main.cpp)
target_link_libraries(${EXEC} PUBLIC Coyote)
target_link_directories(${EXEC} PUBLIC /usr/local/lib)

find_package(Boost REQUIRED COMPONENTS program_options)
target_link_libraries(${EXEC} PUBLIC Boost::program_options)

set_property(TARGET ${EXEC} PROPERTY HIP_ARCHITECTURES ${AMD_GPU})


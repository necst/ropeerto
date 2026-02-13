######################################################################################
# This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
# 
# MIT Licence
# Copyright (c) 2025-2026, Systems Group, ETH Zurich
# All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
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
set(CYT_DIR ${CMAKE_SOURCE_DIR}/../../../Coyote)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteHW REQUIRED)
project(test)


# --------------------------------------------------------
# Auto-generate mutual_info.hpp at CMake configure time
# --------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Generator inputs (override from CLI: -DMI_IN_DIM=... etc.)
set(MI_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/hls/mutual_information_master/" CACHE PATH "Where mutual_info.hpp will be generated")

set(MI_HISTOTYPE "float" CACHE STRING "MI histotype: float|fixed")
set(MI_PE_NUMBER 1 CACHE STRING "Joint histogram PE number")
set(MI_IN_BITS 8 CACHE STRING "Input image bits")
set(MI_IN_DIM 512 CACHE STRING "Input image dimension")
set(MI_BIN_VAL 0 CACHE STRING "Binning reduction value")
set(MI_ENTR_ACC_SIZE 8 CACHE STRING "Entropy accumulator size")
set(MI_PE_ENTROPY 1 CACHE STRING "Entropy PE number")
set(MI_N_COUPLES_MAX 1 CACHE STRING "Max number of couples")
set(MI_PIXELS_PER_READ 32 CACHE STRING "Pixels per read")
set(MI_INTERP_PE_NUMBER 1 CACHE STRING "Interpolator PE number")

option(MI_VITIS "Generate Vitis version" OFF)
option(MI_CACHE_MEM "Enable caching version" OFF)
option(MI_USE_URAM "Use URAM (only meaningful with caching)" OFF)

# Ensure output directory exists
file(MAKE_DIRECTORY "${MI_OUT_DIR}")

# Build the generator command line
set(_mi_gen_args
  "--out_path" "${MI_OUT_DIR}"
  "--histotype" "${MI_HISTOTYPE}"
  "--pe_number" "${MI_PE_NUMBER}"
  "--in_bits" "${MI_IN_BITS}"
  "--in_dim" "${MI_IN_DIM}"
  "--bin_val" "${MI_BIN_VAL}"
  "--entr_acc_size" "${MI_ENTR_ACC_SIZE}"
  "--pe_entropy" "${MI_PE_ENTROPY}"
  "--n_couples_max" "${MI_N_COUPLES_MAX}"
  "--pixels_per_read" "${MI_PIXELS_PER_READ}"
  "--interpolator_pe_number" "${MI_INTERP_PE_NUMBER}"
)

if(MI_VITIS)
  list(APPEND _mi_gen_args "--vitis")
endif()
if(MI_CACHE_MEM)
  list(APPEND _mi_gen_args "--cache_mem")
endif()
if(MI_USE_URAM)
  list(APPEND _mi_gen_args "--use_uram")
endif()

# Run generator now (configure-time)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generator.py" ${_mi_gen_args}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RESULT_VARIABLE _mi_gen_rc
  OUTPUT_VARIABLE _mi_gen_out
  ERROR_VARIABLE  _mi_gen_err
)

if(NOT _mi_gen_rc EQUAL 0)
  message(FATAL_ERROR
    "mutual_info.hpp generation failed.\n"
    "Command: ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generator.py ${_mi_gen_args}\n"
    "stdout:\n${_mi_gen_out}\n"
    "stderr:\n${_mi_gen_err}\n"
  )
endif()

message(STATUS "Generated mutual_info.hpp in: ${MI_OUT_DIR}/mutual_info.hpp")


# Enables three data streams from host memory 
set(EN_STRM 1)
set(N_STRM_AXI 3)

## Confirm that the selected options are allowed
validation_checks_hw()

# Load a user application in Configuration #0, Region #0
load_apps (
    VFPGA_C0_0 "src"
)

# Create the hardware project
create_hw()
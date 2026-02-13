/******************************************
* MIT License
*
* Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide
Conficconi, Eleonora D'Arnese
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.

*MIT License
*
*Copyright (c) [2019] [Davide Conficconi, Eleonora D'Arnese, Marco Domenico
Santambrogio]
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/
/***************************************************************
 *
 * main of the whole application
 * credits goes also to the author of this repo:
 *https://github.com/mariusherzog/ImageRegistration
 *
 ****************************************************************/

//#define HW_REG 1

#ifndef HW_REG
#include "irg_app/include/software_mi/software_mi.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "HIPRigidWarp3D/src/utils/images_io.h"
#include "constants.h" // DIMENSION, HIST_PE, J_HISTO_ROWS, J_HISTO_COLS
#include "irg_app/HAL/HardwareAbstractionLayer.h"
#include "irg_app/app/imagefusion.hpp"
#include "irg_app/core/fusion_algorithms.hpp"
#include "irg_app/core/register_algorithms.hpp"

#define DEVICE_ID 0

using namespace cv;
using namespace std;
using namespace std::placeholders;

std::ostream &bold_on(std::ostream &os);
std::ostream &bold_off(std::ostream &os);

void getBackwardSplit(size_t size, char *string,
                      char *dst) { //, char * prefix){
  int i;
  int j;
  for (i = size; i >= 0; i--) {
    if (string[i] == '.')
      j = i;
    if (string[i] == '/')
      break;
  }
  strncpy(dst, string + i + 1, (size - j - 1));
  dst[size - j - 1] = '\0';
}

void getFinalName(char *im1name, char *im2name, char *finalname) {
  size_t im1size = strlen(im1name);
  size_t im2size = strlen(im2name);
  strcat(finalname, im1name);
  strcat(finalname, im2name);
  const char *format = ".jpeg";
#ifdef HW_REG
  const char *currplat = "-hw";

#else
  const char *currplat = "-sw";

#endif

  strcat(finalname, currplat);
  strcat(finalname, format);
}

int main(int argc, char **argv) {
#ifdef COYOTE_MODE
  std::cout << "COYOTE_MODE" << std::endl;

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <vfpga_id> <pet_path> <ct_path> <out_path> [<depth>] "
                 "[<rangeX>] [<rangeY>] "
                 "[<rangeZ>] [<runs>]"
              << std::endl;
    return 1;
  }

  //  1) Read the vfpga_id
  int vfpga_id = atoi(argv[1]);
#else

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <xclbin_path> <pet_path> <ct_path> <out_path> [<depth>] "
                 "[<rangeX>] "
                 "[<rangeY>] [<rangeZ>] [<runs>] [<gpu_id>]"
              << std::endl;
    return 1;
  }

  std::string xclbin_path = argv[1];
#endif

  std::string pet_path = argv[2];
  std::string ct_path = argv[3];
  std::string out_path = argv[4];
  int depth = argc >= 5 ? atoi(argv[5]) : 246;
  int rangeX = argc >= 6 ? atoi(argv[6]) : 256;
  int rangeY = argc >= 7 ? atoi(argv[7]) : 256;
  float rangeAngZ = argc >= 8 ? atof(argv[8]) : 1.0;
  int runs = argc >= 9 ? atoi(argv[9]) : 1;
  int gpu_id = argc >= 10 ? atoi(argv[10]) : 0;

  const int padding = 0;
  std::cout << "Number of couples: " << depth << std::endl;
  std::cout << "RangeX: " << rangeX << std::endl;
  std::cout << "RangeY: " << rangeY << std::endl;
  std::cout << "RangeAngZ: " << rangeAngZ << std::endl;
  std::cout << "Number of couples: " << depth << std::endl;
  auto available_fusion_names = imagefusion::fusion_strategies();
  auto available_register_names = imagefusion::register_strategies();
  std::cout << "REF path: " << ct_path << std::endl;
  std::cout << "FLOAT path: " << pet_path << std::endl;
  std::cout << "GPU id: " << gpu_id << std::endl;
  file_repository files(ct_path, pet_path);
  std::vector<cv::Mat> reference_image = files.reference_image_3d(depth);
  std::vector<cv::Mat> floating_image = files.floating_image_3d(depth);

#ifdef HW_REG

  std::cout << "HW_REG" << std::endl;
  //------------------------------------------------LOADING
  // XCLBIN------------------------------------------
  // Load xclbin

#ifdef COYOTE_MODE

  std::cout << "COYOTE_MODE" << std::endl;
  device dev = {
    device_index : DEVICE_ID,
    vfpga_index : vfpga_id,
    gpu_index : gpu_id,
    p2p_mode : true
  };

#else

  std::cout << "XRT_MODE" << std::endl;
  device dev = {
    xclbin_path : xclbinPath,
    kernel_name : "mutual_information_master",
    device_index : DEVICE_ID
  };

#endif

  hipSetDevice(gpu_id);

  // array for execution times
  std::vector<double> execution_times(runs, 0.0);

  RigidWarpXYPlane transform;
  HardwareAbstractionLayer board(dev, DIMENSION, depth, transform);

  std::ofstream timing_file("p2p_image_registration.csv", std::ios::app);
  if (!timing_file.is_open()) {
    std::cerr << "Failed to open timing file for appending." << std::endl;
  }
  timing_file << "time\n";

  for (int i = 0; i < runs; i++) {
    board.load_ref(ct_path);
    board.load_flt(pet_path);

    double execution_time = imagefusion::perform_fusion_from_files_3d(
        reference_image, floating_image, "mutualinformation", "alphablend",
        board, rangeX, rangeY, rangeAngZ);
    execution_times.push_back(execution_time);
    std::cout << "Execution time for run " << i + 1 << ": " << execution_time
              << " seconds" << std::endl;
    timing_file << execution_time << "\n";
    timing_file.flush();
  }

  // compute average execution time:
  double average_execution_time =
      std::accumulate(execution_times.begin(), execution_times.end(), 0.0) /
      runs;
  std::cout << "Average execution time over " << runs
            << " runs: " << average_execution_time << " seconds" << std::endl;

  std::cout << "Number of registration steps: " << board.counter << std::endl;

  write_volume_to_file(board.ptr_out, DIMENSION, depth, 0, padding, out_path);
  std::cout << "Saving Volumes" << std::endl;

#else

  // array for execution times
  std::vector<double> execution_times(runs, 0.0);

  uint8_t *registered_volume =
      new uint8_t[DIMENSION * DIMENSION * (depth + padding)];
  for (int i = 0; i < runs; i++) {
    double execution_time = imagefusion::perform_fusion_from_files_3d(
        reference_image, floating_image, "mutualinformation", "alphablend",
        depth, padding, rangeX, rangeY, rangeAngZ, registered_volume);
    execution_times.push_back(execution_time);
    std::cout << "Execution time for run " << i + 1 << ": " << execution_time
              << " seconds" << std::endl;
  }

  // compute average execution time:
  double average_execution_time =
      std::accumulate(execution_times.begin(), execution_times.end(), 0.0) /
      runs;
  std::cout << "Average execution time over " << runs
            << " runs: " << average_execution_time << " seconds" << std::endl;

  std::cout << "Saving Volumes" << std::endl;
  write_volume_to_file(registered_volume, DIMENSION, depth, 0, padding,
                       out_path);

#endif
}

std::ostream &bold_on(std::ostream &os) { return os << "\e[1m"; }

std::ostream &bold_off(std::ostream &os) { return os << "\e[0m"; }

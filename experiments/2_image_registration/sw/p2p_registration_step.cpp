#include <hip/hip_runtime.h>

#include <any>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "HIPRigidWarp3D/src/hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hpp"
#include "HIPRigidWarp3D/src/utils/args_parser.h"
#include "HIPRigidWarp3D/src/utils/images_io.h"
#include "cThread.hpp" // Coyote thread

#define DEFAULT_VFPGA_ID 0

#include "constants.h" // DIMENSION, HIST_PE, J_HISTO_ROWS, J_HISTO_COLS

double software_mi(int n_couples, uint8_t *input_ref, uint8_t *input_flt) {
  double j_h[J_HISTO_ROWS][J_HISTO_COLS];
  for (int i = 0; i < J_HISTO_ROWS; i++) {
    for (int j = 0; j < J_HISTO_COLS; j++) {
      j_h[i][j] = 0.0;
    }
  }

  const int N_COUPLES_TOTAL = n_couples;

  for (int k = 0; k < N_COUPLES_TOTAL; k++) {
    for (int i = 0; i < DIMENSION; i++) {
      for (int j = 0; j < DIMENSION; j++) {
        unsigned int a = input_ref[i * DIMENSION * (N_COUPLES_TOTAL) +
                                   j * (N_COUPLES_TOTAL) + k];
        unsigned int b = input_flt[i * DIMENSION * (N_COUPLES_TOTAL) +
                                   j * (N_COUPLES_TOTAL) + k];
        j_h[a][b] = (j_h[a][b]) + 1;
      }
    }
  }

  for (int i = 0; i < J_HISTO_ROWS; i++) {
    for (int j = 0; j < J_HISTO_COLS; j++) {
      j_h[i][j] = j_h[i][j] / ((N_COUPLES_TOTAL)*DIMENSION * DIMENSION);
    }
  }

  float entropy = 0.0;
  for (int i = 0; i < J_HISTO_ROWS; i++) {
    for (int j = 0; j < J_HISTO_COLS; j++) {
      float v = j_h[j][i];
      if (v > 0.000000000000001) {
        entropy += v * log2(v);
      }
    }
  }
  entropy *= -1;

  double href[ANOTHER_DIMENSION];
  for (int i = 0; i < ANOTHER_DIMENSION; i++) {
    href[i] = 0.0;
  }

  for (int i = 0; i < ANOTHER_DIMENSION; i++) {
    for (int j = 0; j < ANOTHER_DIMENSION; j++) {
      href[i] += j_h[i][j];
    }
  }

  double hflt[ANOTHER_DIMENSION];
  for (int i = 0; i < ANOTHER_DIMENSION; i++) {
    hflt[i] = 0.0;
  }

  for (int i = 0; i < J_HISTO_ROWS; i++) {
    for (int j = 0; j < J_HISTO_COLS; j++) {
      hflt[i] += j_h[j][i];
    }
  }

  double eref = 0.0;
  for (int i = 0; i < ANOTHER_DIMENSION; i++) {
    if (href[i] > 0.000000000001) {
      eref += href[i] * log2(href[i]);
    }
  }
  eref *= -1;

  double eflt = 0.0;
  for (int i = 0; i < ANOTHER_DIMENSION; i++) {
    if (hflt[i] > 0.000000000001) {
      eflt += hflt[i] * log2(hflt[i]);
    }
  }
  eflt = eflt * (-1);

  double mutualinfo = eref + eflt - entropy;
  return mutualinfo;
}

void compute_mi(coyote::cThread &coyote_thread, uint8_t *input_flt,
                uint8_t *input_ref, float *mutual_info, uint64_t *n_couples_mem,
                std::ofstream &timing_file) {
  uint32_t local_write_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE);
  // std::cout << "Local write count: " << local_write_count << std::endl;
  uint32_t local_read_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ);

  size_t elems = DIMENSION * DIMENSION * n_couples_mem[0];
  uint32_t bytes = elems * sizeof(uint8_t);

  // Set Coyote thread arguments
  coyote::localSg sg_flt;

  memset(&sg_flt, 0, sizeof(coyote::localSg));
  sg_flt = {.addr = input_flt, .len = bytes, .dest = 0};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_flt);

  // std::cout << "Floating volume written to Coyote thread" << std::endl;
  coyote::localSg sg_ref;
  memset(&sg_ref, 0, sizeof(coyote::localSg));
  sg_ref = {.addr = input_ref, .len = bytes, .dest = 1};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_ref);
  // std::cout << "Reference volume written to Coyote thread" << std::endl;

  // Set the number of couples in the kernel
  coyote::localSg sg_n_couples;
  memset(&sg_n_couples, 0, sizeof(coyote::localSg));
  sg_n_couples = {.addr = n_couples_mem, .len = sizeof(uint64_t), .dest = 2};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_n_couples);
  // std::cout << "Number of couples set in Coyote thread" << std::endl;

  coyote_thread.setCSR(static_cast<uint64_t>(0x1), static_cast<uint32_t>(0));
  // std::cout << "Control register set in Coyote thread" << std::endl;

  coyote::localSg sg_out;
  memset(&sg_out, 0, sizeof(coyote::localSg));
  sg_out = {.addr = mutual_info, .dest = 0, .len = sizeof(float)};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_WRITE, sg_out);
  std::chrono::high_resolution_clock::time_point time_start =
      std::chrono::high_resolution_clock::now();
  while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) <=
         local_write_count)
    ;
  std::chrono::high_resolution_clock::time_point time_end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = time_end - time_start;
  // timing_file << elapsed.count() << ",";
  //  std::cout << "check completed: "
  //            <<
  //            coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_WRITE)
  //            << std::endl;
  //  std::cout << "Mutual information computed in Coyote thread" << std::endl;
}

void compare_and_save_mi(float mi_sw, float mi_hw) {
  // Compare software and hardware MI values
  if (std::abs(mi_sw - mi_hw) > 1e-3) {
    std::cerr << "Error: Software and hardware MI results do not match!\n";
    std::cerr << "Software MI: " << mi_sw << "\n";
    std::cerr << "Hardware MI: " << mi_hw << "\n";
  } else {
    std::cout << "Software and hardware MI results match!\n";
  }
}

int main(int argc, char **argv) {
  if (argc < 10) {
    std::cerr << "Usage: " << argv[0]
              << " <vfpga_id> <PET_folder> <CT_folder> <out_folder>"
                 " <tx> <ty> <ang> <runs> <gpu_id> [depth]\n";
    return 1;
  }

  int vfpga_id = atoi(argv[1]);
  // 2) Input/output folders and transformation parameters
  std::string pet_dir = argv[2];
  std::string ct_dir = argv[3];
  std::string out_dir = argv[4];
  float tx = std::stof(argv[5]);
  float ty = std::stof(argv[6]);
  float ang = std::stof(argv[7]);
  int runs = std::atoi(argv[8]);
  int gpu_id = std::atoi(argv[9]);

  // write CSV file to save timing
  std::ofstream timing_file("./p2p_registration_step.csv", std::ios::app);
  if (!timing_file) {
    std::cerr << "Error: Could not open timing file for writing\n";
    return 1;
  }
  timing_file << "time\n";

  int depth = 246;
  if (argc >= 11) {
    depth = std::atoi(argv[10]);
    if (depth <= 0) {
      std::cerr << "Error: depth must be > 0\n";
      return 1;
    }
  }
  size_t elems = DIMENSION * DIMENSION * depth;
  uint32_t bytes = elems * sizeof(uint8_t);

  RigidWarpXYPlane hip_transform;

  hipSetDevice(gpu_id);
  std::cout << "Warming up HIP kernel...\n";
  for (int i = 0; i < 10; i++)
    hip_transform.run(0, 0, 0);

  std::cout << "Allocating memory for volumes...\n";
  uint8_t *float_cpu =
      (uint8_t *)malloc(DIMENSION * DIMENSION * depth * sizeof(uint8_t));

  uint64_t n_couples_cpu = depth;

  coyote::cThread coyote_thread(vfpga_id, getpid(), 0);
  uint8_t *flt = (uint8_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::GPU, bytes, false, (uint32_t)gpu_id});
  uint8_t *ref =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, bytes});
  uint8_t *out = (uint8_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::GPU, bytes, false, (uint32_t)gpu_id});
  float *mutual_info = (float *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, 1 * sizeof(float)});

  uint64_t *n_couples_mem = (uint64_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, sizeof(uint64_t)});

  if (!flt || !ref || !out || !mutual_info || !n_couples_mem) {
    throw std::runtime_error(
        "Could not allocate memory for vectors, exiting...");
  }
  hipPointerAttribute_t attr;

  hipError_t status = hipPointerGetAttributes(&attr, flt);
  if (status != hipSuccess) {
    std::cerr << "hipPointerGetAttributes failed: " << hipGetErrorString(status)
              << std::endl;
  } else {
    std::cout << "Pointer attributes for flt:\n";
    std::cout << "  devicePointer: " << attr.devicePointer << "\n";
    std::cout << "  hostPointer:   " << attr.hostPointer << "\n";
  }

  // 1) Load volumes
  std::cout << "Loading volumes...\n";
  read_volume_from_folder(float_cpu, DIMENSION, depth, pet_dir);
  read_volume_from_folder(ref, DIMENSION, depth, ct_dir);

  hip_transform.moveToGPU(flt, float_cpu, DIMENSION, depth);
  std::cout << "Warming up HIP kernel...\n";
  for (int i = 0; i < 10; i++) {
    hip_transform.run_external(flt, out, tx, ty, ang, DIMENSION, depth);
  }

  // array of times for each run, to average
  std::vector<double> times(runs, 0.0);
  for (int i = 0; i < runs; i++) {

    std::cout << "Running HIP warp...\n";


    *n_couples_mem = static_cast<uint64_t>(depth);

    // 2) Running Hip Warp
    std::chrono::high_resolution_clock::time_point time_start =
        std::chrono::high_resolution_clock::now();
    hip_transform.moveToGPU(flt, float_cpu, DIMENSION, depth);

    double t =
        hip_transform.run_external(flt, out, tx, ty, ang, DIMENSION, depth);

    compute_mi(coyote_thread, out, ref, mutual_info, n_couples_mem,
               timing_file);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(time_end - time_start);

    times[i] = elapsed.count(); // Store the time for this run

    float mi_value = 0.0f;
    mi_value = mutual_info[0];
    std::cout << "Mutual Information (warped vs CT): " << mi_value << "\n";

    hip_transform.moveFromGPU(float_cpu, out, DIMENSION, depth);

    // software comparison MI
    std::cout << "Computing Mutual Information (software)...\n";
    float sw_mi = software_mi(depth, ref, float_cpu);
    std::cout << "Software Mutual Information (warped vs CT): " << sw_mi
              << "\n";
    compare_and_save_mi(sw_mi, mi_value);
    timing_file << times[i] << "\n";
    timing_file.flush();
  }
  // Calculate average time
  double avg_time = 0.0;
  for (const auto &time : times) {
    avg_time += time;
  }
  avg_time /= runs;
  std::cout << "Average execution time over " << runs << " runs: " << avg_time
            << " s\n";

  coyote_thread.userUnmap((void *)flt);
  coyote_thread.userUnmap((void *)ref);
  coyote_thread.userUnmap((void *)out);
  coyote_thread.userUnmap((void *)mutual_info);
  coyote_thread.userUnmap((void *)n_couples_mem);

  save_volume_into_folder(float_cpu, DIMENSION, depth, out_dir);
  free(float_cpu);

  timing_file.close();
  return 0;
}
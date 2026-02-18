#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "HIPRigidWarp3D/src/hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hpp"
#include "HIPRigidWarp3D/src/utils/args_parser.h"
#include "HIPRigidWarp3D/src/utils/images_io.h"

#include "irg_app/include/software_mi/software_mi.hpp"

#ifdef COYOTE_MODE
#include <any>
#include "cThread.hpp" // Coyote thread
#else
// XRT / AIE includes
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

#endif

#include "constants.h" // DIMENSION, HIST_PE, J_HISTO_ROWS, J_HISTO_COLS

void compare_and_save_mi(float mi_sw, float mi_hw, float tx, float ty,
                         float ang) {
  // Compare software and hardware MI values
  if (std::abs(mi_sw - mi_hw) > 1e-3) {
    std::cerr << "Error: Software and hardware MI results do not match!\n";
    std::cerr << "Software MI: " << mi_sw << "\n";
    std::cerr << "Hardware MI: " << mi_hw << "\n";
  } else {
    std::cout << "Software and hardware MI results match!\n";
  }
  // Save the results to a file
  std::ofstream hw_mi_file("mi_results.csv", std::ios::app);
  if (!hw_mi_file) {
    std::cerr << "Error opening hardware MI file for writing\n";
    return;
  }
  hw_mi_file << mi_sw << "," << mi_hw << "," << tx << "," << ty << "," << ang
             << "\n";
  hw_mi_file.close();
}

// HIP GPU capability printer
void printGPUCapabilities_HIP() {
  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
    std::cerr << "Errore recupero device count HIP\n";
    return;
  }
  for (int d = 0; d < deviceCount; ++d) {
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, d);
    std::cout << "GPU " << d << ": " << p.name
              << ", SMs=" << p.multiProcessorCount
              << ", maxThreadsBlk=" << p.maxThreadsPerBlock << "\n";
  }
}

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

#ifdef COYOTE_MODE

float compute_mi(coyote::cThread &coyote_thread, uint8_t *input_flt,
                 uint8_t *input_ref, float *mutual_info,
                 uint64_t *n_couples_mem) {
  uint32_t local_write_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE);
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
  sg_out = {.addr = mutual_info, .len = sizeof(float), .dest = 0};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_WRITE, sg_out);

  while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) <=
         local_write_count)
    ;
  // std::cout << "Mutual information computed in Coyote thread" << std::endl;

  // Read back the result

  return mutual_info[0];
}

#else

// =============================================================================
// Implementation of compute_mi: hardware-accelerated mutual information
// =============================================================================
float compute_mi(xrt::device &device, xrt::kernel &krnl, xrt::bo &bo_ref,
                 xrt::bo &bo_flt, xrt::bo &bo_out, const uint8_t *input_flt,
                 int n_couples) {
  int total = n_couples;
  size_t elems = DIMENSION * DIMENSION * total;
  size_t bytes = elems * sizeof(uint8_t);

  // Set kernel arguments
  auto run = xrt::run(krnl);
  run.set_arg(0, bo_flt);
  run.set_arg(1, bo_ref);
  run.set_arg(2, bo_out);
  run.set_arg(3, total);
  run.set_arg(4, 0);

  // Transfer floating volume to device, execute, and read back result
  bo_flt.write(input_flt);
  bo_flt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run.start();
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  float mi;
  bo_out.read(&mi);
  return mi;
}

#endif

bool get_xclbin_path(std::string &out) {
  if (auto *m = std::getenv("XCL_EMULATION_MODE")) {
    if (std::string(m) == "hw_emu")
      out = "overlay_hw_emu.xclbin";
    else
      return false;
  } else {
    out = "overlay_hw.xclbin";
  }
  return true;
}

#define DEFAULT_VFPGA_ID 0

int main(int argc, char **argv) {
  std::cout << "NON P2P Registration Step\n";
  float tx, ty, ang;
#ifdef COYOTE_MODE

  if (argc < 10) {
    std::cerr << "Usage: " << argv[0]
              << " <vfpga_id> <PET_folder> <CT_folder> <out_folder>"
                 " <tx> <ty> <ang> <runs> <gpu_id> [depth]\n";
    return 1;
  }

  int vfpga_id = atoi(argv[1]);
#else
  if (argc < 10) {
    std::cerr << "Usage: " << argv[0]
              << " <vfpga_id> <PET_folder> <CT_folder> <out_folder>"
                 " <tx> <ty> <ang> <runs> <gpu_id> [depth]\n";
    return 1;
  }
  // 1) Read the path to the bitstream
  std::string xclbin_path = argv[1];
#endif

  // 2) Input/output folders and transformation parameters
  std::string pet_dir = argv[2];
  std::string ct_dir = argv[3];
  std::string out_dir = argv[4];
  float user_tx = std::stof(argv[5]);
  float user_ty = std::stof(argv[6]);
  float user_ang = std::stof(argv[7]);
  int runs = std::atoi(argv[8]);
  int gpu_id = std::atoi(argv[9]);
  std::cout << "GPU ID: " << gpu_id << "\n";

  hipSetDevice(gpu_id);

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

  // 2) HIP transform
  printGPUCapabilities_HIP();

  RigidWarpXYPlane hip_transform(gpu_id);
  std::cout << "Warming up HIP kernel...\n";

  for (int i = 0; i < 10; i++)
    hip_transform.run(0, 0, 0);
  std::cout << "Running " << runs
            << " iterations with random transformations...\n";

#ifdef COYOTE_MODE

  coyote::cThread coyote_thread(vfpga_id, getpid(), 0);
  uint8_t *flt =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, bytes});
  uint8_t *ref =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, bytes});
  uint8_t *out =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, bytes});
  float *mutual_info = (float *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, 16 * sizeof(float)});
  uint64_t *n_couples_mem = (uint64_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, sizeof(uint64_t)});
  if (!flt || !ref || !out || !mutual_info || !n_couples_mem) {
    throw std::runtime_error(
        "Could not allocate memory for vectors, exiting...");
  }

  std::cout << "Buffers allocated" << std::endl;
  n_couples_mem[0] = (uint64_t)depth;
  std::cout << "Number of couples: " << n_couples_mem[0] << std::endl;

  // 1) Load volumes
  std::cout << "Loading PET volume...\n";
  read_volume_from_folder(flt, DIMENSION, depth, pet_dir);
  std::cout << "Loading CT reference...\n";
  read_volume_from_folder(ref, DIMENSION, depth, ct_dir);

  std::ofstream timing_file("nop2p_registration_step.csv", std::ios::app);
  if (!timing_file) {
    std::cerr << "Error opening timing file for writing\n";
  }
  timing_file << "time\n";

  // array of times for each run, to average
  std::vector<double> times(runs, 0.0);

  for (int i = 0; i < runs; i++) {

      tx = user_tx;   // Use user-defined tx
      ty = user_ty;   // Use user-defined ty
      ang = user_ang; // Use user-defined angle


    // 2) Hip Transform
    std::cout << "Running HIP warp...\n";

    // measure time with precision
    std::chrono::high_resolution_clock::time_point time_start =
        std::chrono::high_resolution_clock::now();
    hip_transform.transferToGPU(flt, DIMENSION, depth);
    double t = hip_transform.run(tx, ty, ang);
    // std::cout << "HIP exec time: " << t << " s\n";
    hip_transform.transferFromGPU(out);
    hipError_t err = hipDeviceSynchronize();

    if (err != hipSuccess) {
      std::cerr << "Sync Error: " << hipGetErrorString(err) << std::endl;
      return 1;
    }

    // std::cout << "Computing Mutual Information...\n";
    float mi = compute_mi(coyote_thread, out, ref, mutual_info, n_couples_mem);

    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;
    std::cout << "Registration step exec time: " << elapsed.count() << " s\n";
    times[i] = elapsed.count(); // Store the time for this run

    std::cout << "Mutual Information (warped vs CT): " << mi << "\n";

    // software comparison MI
    std::cout << "Computing Mutual Information (software)...\n";
    float sw_mi = software_mi(depth, out, ref);
    std::cout << "Software Mutual Information (warped vs CT): " << sw_mi
              << "\n";

    // Compare software and hardware MI values
    compare_and_save_mi(sw_mi, mi, tx, ty, ang);

    timing_file << elapsed.count() << "\n";
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
  std::cout << "Saving warped volume...\n";

  coyote_thread.userUnmap((void *)flt);
  coyote_thread.userUnmap((void *)ref);
  coyote_thread.userUnmap((void *)out);
  coyote_thread.userUnmap((void *)mutual_info);
  coyote_thread.userUnmap((void *)n_couples_mem);

  save_volume_into_folder(flt, DIMENSION, depth, out_dir);

#else

  // 1) XRT / AIE mutual information
  xrt::device device(0);
  auto uuid = device.load_xclbin(xclbin_path);
  xrt::kernel krnl(device, uuid, "mutual_information_master");

  // 2) Load volumes

  uint8_t *in_vol = new uint8_t[elems];
  uint8_t *out_vol = new uint8_t[elems];
  uint8_t *ref_vol = new uint8_t[elems];

  std::cout << "Loading PET volume...\n";
  read_volume_from_folder(in_vol, size, depth, pet_dir);
  std::cout << "Loading CT reference...\n";
  read_volume_from_folder(ref_vol, size, depth, ct_dir);

  xrt::bo bo_ref(device, bytes, xrt::bo::flags::normal, krnl.group_id(1));
  xrt::bo bo_flt(device, bytes, xrt::bo::flags::normal, krnl.group_id(0));
  xrt::bo bo_out(device, sizeof(float), xrt::bo::flags::normal,
                 krnl.group_id(2));

  for (int i = 0; i < runs; i++) {
    if (i == 0) {
      tx = user_tx;   // Use user-defined tx
      ty = user_ty;   // Use user-defined ty
      ang = user_ang; // Use user-defined angle
    } else {
      // Generate random tx, ty, ang for each run
      srand(static_cast<unsigned int>(time(0) +
                                      i)); // Seed with time and run index
      // genera TX e TY randomicamente tra - 50 e 50
      tx =
          static_cast<float>(rand() % 100 - 50); // Random tx between -50 and 50
      ty =
          static_cast<float>(rand() % 100 - 50); // Random ty between -50 and 50
      ang = static_cast<float>(rand() % 30 + 10) /
            100.0f; // Random angle between 0.1 and 0.4
    }
    // 3. Transform
    std::cout << "Running HIP warp...\n";
    hip_transform.transferToGPU(in_vol, size, depth);
    double t = hip_transform.run(tx, ty, ang);
    std::cout << "HIP exec time: " << t << " s\n";
    hip_transform.transferFromGPU(out_vol);

    // Transfer reference volume to device
    bo_ref.write(ref_vol);
    bo_ref.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 4. Compute MI Hardware
    std::cout << "Computing Mutual Information...\n";
    float mi = compute_mi(device, krnl, bo_ref, bo_flt, bo_out, out_vol, depth);
    std::cout << "Mutual Information (warped vs CT): " << mi << "\n";

    // 5. Compute MI Software
    std::cout << "Computing Mutual Information (software)...\n";
    float sw_mi = software_mi(depth, out_vol, ref_vol);
    std::cout << "Software Mutual Information (warped vs CT): " << sw_mi
              << "\n";
    // 3) Save warped volume
    compare_and_save_mi(sw_mi, mi, tx, ty, ang);
    float diff = std::abs(mi - sw_mi);
    save_volume_into_folder(out_vol, DIMENSION, depth, out_dir);
  }

  delete[] in_vol;
  delete[] out_vol;
  delete[] ref_vol;
#endif

  return 0;
}

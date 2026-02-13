#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "HIPRigidWarp3D/src/utils/images_io.h" // read_volume_from_folder()
#ifdef COYOTE_MODE
#include <any>

#include "cThread.hpp" // Coyote thread
#else
// XRT / AIE includes
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

#endif
#include "constants.h" // DIMENSION, HIST_PE, J_HISTO_ROWS, J_HISTO_COLS

#define DEFAULT_DEVICE_ID 0

// =============================================================================
// Implementation of compute_mi: hardware-accelerated mutual information
// =============================================================================

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

#ifdef COYOTE_MODE

float compute_mi(coyote::cThread &coyote_thread, uint8_t *input_flt,
                 uint8_t *input_ref, float *mutual_info,
                 uint64_t *n_couples_mem) {
  uint32_t local_write_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE);
  uint32_t local_read_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ);
  // Add padding so total is multiple of HIST_PE
  size_t elems = DIMENSION * DIMENSION * n_couples_mem[0];
  uint32_t bytes = elems * sizeof(uint8_t);

  // Set Coyote thread arguments
  coyote::localSg sg_flt;
  memset(&sg_flt, 0, sizeof(coyote::localSg));
  sg_flt = {.addr = input_flt, .len = bytes, .dest = 0};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_flt);

  std::cout << "Floating volume written to Coyote thread" << std::endl;
  coyote::localSg sg_ref;
  memset(&sg_ref, 0, sizeof(coyote::localSg));
  sg_ref = {.addr = input_ref, .len = bytes, .dest = 1};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_ref);
  std::cout << "Reference volume written to Coyote thread" << std::endl;

  // Set the number of couples in the kernel
  coyote::localSg sg_n_couples;
  memset(&sg_n_couples, 0, sizeof(coyote::localSg));
  sg_n_couples = {.addr = n_couples_mem, .len = sizeof(uint64_t), .dest = 2};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_n_couples);
  std::cout << "Number of couples set in Coyote thread" << std::endl;

  std::cout << "Number of Local reads: "
            << coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ)
            << std::endl;

  coyote_thread.setCSR(static_cast<uint64_t>(0x1), static_cast<uint32_t>(0));

  std::cout << "Control register set in Coyote thread" << std::endl;

  coyote::localSg sg_out;
  memset(&sg_out, 0, sizeof(coyote::localSg));
  sg_out = {.addr = mutual_info, .len = sizeof(float), .dest = 0};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_WRITE, sg_out);

  std::cout << "Number of Writes: "
            << coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE)
            << std::endl;
  while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) <=
         local_write_count)
    ;
  std::cout << "Mutual information computed in Coyote thread" << std::endl;

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
  // Add padding so total is multiple of HIST_PE
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

// =============================================================================
// Implementation of software_mi: pure-CPU mutual information
// =============================================================================
double software_mi(int n_couples, const uint8_t *input_ref,
                   const uint8_t *input_flt) {
  const int padding = 0;
  int N = n_couples + padding;

  // Build joint histogram j_h[a][b]
  double j_h[J_HISTO_ROWS][J_HISTO_COLS] = {{0.0}};
  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < DIMENSION; ++i) {
      for (int j = 0; j < DIMENSION; ++j) {
        int idx = i * DIMENSION * N + j * N + k;
        unsigned a = input_ref[idx];
        unsigned b = input_flt[idx];
        j_h[a][b] += 1.0;
      }
    }
  }

  // Normalize histogram to probabilities
  double norm = double(N) * DIMENSION * DIMENSION;
  for (int i = 0; i < J_HISTO_ROWS; ++i) {
    for (int j = 0; j < J_HISTO_COLS; ++j) {
      j_h[i][j] /= norm;
    }
  }

  // Compute joint entropy H(X,Y)
  double Hxy = 0.0;
  for (int i = 0; i < J_HISTO_ROWS; ++i) {
    for (int j = 0; j < J_HISTO_COLS; ++j) {
      double p = j_h[i][j];
      if (p > 1e-15) {
        Hxy -= p * std::log2(p);
      }
    }
  }

  // Compute marginal distributions px and py
  double px[J_HISTO_ROWS] = {0.0}, py[J_HISTO_COLS] = {0.0};
  for (int i = 0; i < J_HISTO_ROWS; ++i) {
    for (int j = 0; j < J_HISTO_COLS; ++j) {
      px[i] += j_h[i][j];
      py[j] += j_h[i][j];
    }
  }

  // Compute marginal entropies H(X) and H(Y)
  double Hx = 0.0, Hy = 0.0;
  for (int i = 0; i < J_HISTO_ROWS; ++i) {
    if (px[i] > 1e-15) {
      Hx -= px[i] * std::log2(px[i]);
    }
  }
  for (int j = 0; j < J_HISTO_COLS; ++j) {
    if (py[j] > 1e-15) {
      Hy -= py[j] * std::log2(py[j]);
    }
  }

  // MI = H(X) + H(Y) – H(X,Y)
  return Hx + Hy - Hxy;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  const int n_run = 50;
  const int depth = 246;
  const size_t V = DIMENSION * DIMENSION * depth;
  const size_t bytes = V * sizeof(uint8_t);

#ifdef COYOTE_MODE
  // Coyote mode: use cThread for parallel execution
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <vfpga_id> <PET_folder> <CT_folder>\n";
    return 1;
  }

  int vfpga_id = atoi(argv[1]);
#else
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <xclbin_path> <PET_folder> <CT_folder>\n";
    return 1;
  }
  std::string xclbin_path = argv[1];

#endif

  // Parse command-line arguments
  std::string pet_dir = argv[2];
  std::string ct_dir = argv[3];

#ifdef COYOTE_MODE

  coyote::cThread coyote_thread(vfpga_id, getpid(), 0);
  // Allocate host buffers for PET and CT volumes
  uint8_t *pet_vol =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, bytes});
  uint8_t *ct_vol =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, bytes});

#else

  // Allocate host buffers for PET and CT volumes
  uint8_t *pet_vol = new uint8_t[V];
  uint8_t *ct_vol = new uint8_t[V];
#endif

#ifdef COYOTE_MODE
  float *mutual_info = (float *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, 1 * sizeof(float)});
  uint64_t *n_couples_mem = (uint64_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, sizeof(uint64_t)});
  if (!pet_vol || !ct_vol || !mutual_info || !n_couples_mem) {
    throw std::runtime_error(
        "Could not allocate memory for vectors, exiting...");
  }
  n_couples_mem[0] = (uint64_t)depth;
  srand(1234);

  for (int run = 0; run < n_run; ++run) {
    int max_noise = rand() % 150;
    if (max_noise < 10)
      max_noise = 10;

    for (size_t i = 0; i < V; ++i) {
      pet_vol[i] = static_cast<uint8_t>(rand() % 256);
      int noise = (rand() % (2 * max_noise + 1)) - max_noise;
      int value = static_cast<int>(pet_vol[i]) + noise;
      ct_vol[i] = static_cast<uint8_t>(std::max(0, std::min(255, value)));
    }
    std::cout << "Computing Mutual Information...\n";
    float mi_hw =
        compute_mi(coyote_thread, pet_vol, ct_vol, mutual_info, n_couples_mem);
    std::cout << "Mutual Information (warped vs CT): " << mi_hw << "\n";
    // 5) (Optional) Compute mutual information in software for validation
    std::cout << "Computing MI in software for comparison...\n";
    double mi_sw = software_mi(depth, pet_vol, ct_vol);
    std::cout << "Software MI: " << mi_sw << "\n";

    // Compare software and hardware MI results
    compare_and_save_mi(mi_sw, mi_hw, 0.0f, 0.0f, 0.0f);
  }
  coyote_thread.userUnmap((void *)pet_vol);
  coyote_thread.userUnmap((void *)ct_vol);
  coyote_thread.userUnmap((void *)mutual_info);
  coyote_thread.userUnmap((void *)n_couples_mem);

#else
  // 2) Initialize XRT/AIE and load bitstream
  xrt::device device(DEFAULT_DEVICE_ID);
  auto uuid = device.load_xclbin(xclbin_path);
  xrt::kernel krnl(device, uuid, "mutual_information_master");

  // apri file csv in scrittura
  std::ofstream myfile;
  myfile.open("mi_result.csv");
  myfile << "exec_time,e2e_time" << std::endl;

  float mi_hw = 0.0;
  xrt::bo bo_ref(device, bytes, xrt::bo::flags::normal, krnl.group_id(1));
  // Allocate BO for the floating volume and for the output MI
  xrt::bo bo_flt(device, bytes, xrt::bo::flags::normal, krnl.group_id(0));
  xrt::bo bo_out(device, sizeof(float), xrt::bo::flags::normal,
                 krnl.group_id(2));
  std::chrono::duration<double> elapsed_exec;

  for (int i = 0; i < n_run; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    bo_ref.write(ct_vol);
    bo_ref.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // 4) Compute mutual information on hardware
    mi_hw = compute_mi(device, krnl, bo_ref, bo_flt, bo_out, pet_vol, depth,
                       elapsed_exec);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_e2e = end - start;
    myfile << elapsed_exec.count() << "," << elapsed_e2e.count() << std::endl;
  }

  myfile.close();

  // 5) (Optional) Compute mutual information in software for validation
  std::cout << "Computing MI in software for comparison...\n";
  double mi_sw = software_mi(depth, pet_vol, ct_vol);

  std::cout << "Software MI: " << mi_sw << "\n";
  compare_and_save_mi(mi_sw, mi_hw, 0.0f, 0.0f, 0.0f);

  delete[] pet_vol;
  delete[] ct_vol;
#endif

  return 0;
}

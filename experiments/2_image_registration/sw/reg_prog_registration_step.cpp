#include <any>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsakmt/hsakmt.h>
#include <iostream>
#include <limits>
#include <string>

#include "HIPRigidWarp3D/src/hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hpp"
#include "HIPRigidWarp3D/src/utils/args_parser.h"
#include "HIPRigidWarp3D/src/utils/images_io.h"
#include "cThread.hpp" // Coyote thread

#define DEFAULT_VFPGA_ID 0

#include "constants.h" // DIMENSION, HIST_PE, J_HISTO_ROWS, J_HISTO_COLS

__device__ void dummy_sleep() {
  int tot = 0;
  while (tot < 100000) {
    printf("tot counter = %d: \n", tot);
    tot++;
  }
}
// constexpr auto const targetRegion = 0;
#define STORE(DST, SRC)                                                        \
  __hip_atomic_store((DST), (SRC), __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM)

//#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

__device__ inline auto setCSR(volatile uint64_t *ctrl_reg, uint64_t val,
                              uint32_t offs) {
  asm volatile("flat_store_dwordx2 %0 %1 glc slc \n"
               :
               : "v"(&ctrl_reg[offs]), "v"(val));
  asm volatile("s_waitcnt lgkmcnt(0)");
  ////
  asm volatile("s_waitcnt vmcnt(0)");
}

#define ATOMIC_STORE_SEQ(DST, SRC)                                             \
  __hip_atomic_store((DST), (SRC), __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT)

__device__ inline auto getCSR(volatile uint64_t *ctrl_reg, uint32_t offs) {

  uint64_t val;
  //"global_load_dwordx2 %0 %1 off glc slc \n"

  asm volatile("flat_load_dwordx2 %0 %1 glc slc \n"
               : "=v"(val)
               : "v"(&ctrl_reg[offs]));
  return val;
}
__device__ int check = 0;

__device__ inline auto setCSR_Atomic_SEQ(volatile uint32_t *ctrl_reg,
                                         uint64_t val, uint32_t offs) {

  // ATOMIC_STORE_SEQ(&ctrl_reg[offs],val);
  STORE(&ctrl_reg[offs], val);
  // __threadfence_system();

  asm volatile("s_waitcnt vmcnt(0)");
}

__global__ void launch_basic_test(volatile uint32_t *ctrl_reg) {
  printf("Hello\n");

  setCSR_Atomic_SEQ(ctrl_reg, 0x1, 0);

  /*
  setCSR_Atomic_SEQ(ctrl_reg, 0xbbbbbbbb, 1);
  setCSR_Atomic_SEQ(ctrl_reg, 0xcccccccc, 2);
  setCSR_Atomic_SEQ(ctrl_reg, 0xd0d0d0d0, 3);
  setCSR_Atomic_SEQ(ctrl_reg, 0xe0e0e0e0, 4);
  setCSR_Atomic_SEQ(ctrl_reg, 0xf0f0f0f0, 5);

  setCSR_Atomic_SEQ(ctrl_reg, 0xaaaaaaaa, 0);
  setCSR_Atomic_SEQ(ctrl_reg, 0xffffffff, 0);

  int value = 3;
  int x = 0xeeeeeeee;
  setCSR(ctrl_reg, x, value);
  */
}

#define CTRL_OPCODE_MASK 0x1f
#define CTRL_STRM_MASK 0x3
#define CTRL_DEST_MASK 0xf
#define CTRL_PID_MASK 0x3f
#define CTRL_VFID_MASK 0xf
#define CTRL_LEN_MASK 0xffffffff
#define CTRL_PID_OFFS (10)
#define CTRL_DEST_OFFS (16)
#define CTRL_LAST (1UL << 20)     // 1048576
#define CTRL_START (1UL << 21)    // 2097152
#define CTRL_CLR_STAT (1UL << 22) // 4194304
#define CTRL_LEN_OFFS (32)

#define CTRL_REG 0
#define VADDR_RD_REG 1
#define CTRL_REG_2 2
#define VADDR_WR_REG 3

/**
 * coper = 0 if read, 1 if write
 */
__global__ void gpu_invoke(volatile uint32_t *ctrl_reg, int coper,
                           uint32_t src_len, uint32_t dst_len,
                           uint32_t src_dest, uint32_t dst_dest, void *src_addr,
                           void *dst_addr, int32_t ctid) {
  int n_sg = 1;

  // Arrays for src + dst address, src + dst ctrl for all entries of the
  // scatter-gather-list
  uint64_t addr_cmd_src[n_sg], addr_cmd_dst[n_sg];
  uint64_t ctrl_cmd_src[n_sg], ctrl_cmd_dst[n_sg];

  // Values for remote and local addr and ctrl
  uint64_t addr_cmd_r, addr_cmd_l;
  uint64_t ctrl_cmd_r, ctrl_cmd_l;

  constexpr unsigned long const MAX_TRANSFER_SIZE = 128 * 1024 * 1024;
  // just to remember that you CANNOT transfer more than 128 MB...

  ctrl_cmd_src[0] =
      // RD
      ((ctid & CTRL_PID_MASK) << CTRL_PID_OFFS) |
      ((src_dest & CTRL_DEST_MASK) << CTRL_DEST_OFFS) |
      ((0 == (n_sg - 1) ? ((1) ? CTRL_LAST : 0x0) : 0x0)) |
      ((1 & CTRL_STRM_MASK) << CTRL_STRM_OFFS) |
      ((coper == 0) ? CTRL_START : 0x0) | (0 ? CTRL_CLR_STAT : 0x0) |
      (static_cast<uint64_t>(src_len) << CTRL_LEN_OFFS);

  addr_cmd_src[0] = reinterpret_cast<uint64_t>(src_addr);

  ctrl_cmd_dst[0] =
      // WR
      ((ctid & CTRL_PID_MASK) << CTRL_PID_OFFS) |
      ((dst_dest & CTRL_DEST_MASK) << CTRL_DEST_OFFS) |
      ((0 == (n_sg - 1) ? ((1) ? CTRL_LAST : 0x0) : 0x0)) |
      ((1 & CTRL_STRM_MASK) << CTRL_STRM_OFFS) |
      ((coper == 1) ? CTRL_START : 0x0) | (0 ? CTRL_CLR_STAT : 0x0) |
      (static_cast<uint64_t>(dst_len) << CTRL_LEN_OFFS);
  addr_cmd_dst[0] = reinterpret_cast<uint64_t>(dst_addr);

  uint64_t offs_3 = addr_cmd_dst[0];
  uint64_t offs_2 = ctrl_cmd_dst[0];
  uint64_t offs_1 = addr_cmd_src[0];
  uint64_t offs_0 = ctrl_cmd_src[0];

  uint32_t offs_32 = (offs_3 >> 32) & 0x00000000ffffffff; // 32 bits
  uint32_t offs_31 = offs_3 & 0x00000000ffffffff;         // 32 bits
  uint32_t offs_22 = (offs_2 >> 32) & 0x00000000ffffffff; // 32 bits
  uint32_t offs_21 = offs_2 & 0x00000000ffffffff;         // 32 bits
  uint32_t offs_12 = (offs_1 >> 32) & 0x00000000ffffffff; // 32 bits
  uint32_t offs_11 = offs_1 & 0x00000000ffffffff;         // 32 bits
  uint32_t offs_02 = (offs_0 >> 32) & 0x00000000ffffffff; // 32 bits
  uint32_t offs_01 = offs_0 & 0x00000000ffffffff;         // 32 bits

  /*
  setCSR_Atomic_SEQ(ctrl_reg, offs_3, VADDR_WR_REG);

  setCSR_Atomic_SEQ(ctrl_reg, offs_2, CTRL_REG_2);

  setCSR_Atomic_SEQ(ctrl_reg, offs_1, VADDR_RD_REG);

  setCSR_Atomic_SEQ(ctrl_reg, offs_0, CTRL_REG);

  */
  setCSR_Atomic_SEQ(ctrl_reg, offs_32, VADDR_WR_REG + 4);
  setCSR_Atomic_SEQ(ctrl_reg, offs_31, VADDR_WR_REG + 3);

  setCSR_Atomic_SEQ(ctrl_reg, offs_21, CTRL_REG_2 + 3);
  setCSR_Atomic_SEQ(ctrl_reg, offs_22, CTRL_REG_2 + 2);

  setCSR_Atomic_SEQ(ctrl_reg, offs_11, VADDR_RD_REG + 2);
  setCSR_Atomic_SEQ(ctrl_reg, offs_12, VADDR_RD_REG + 1);

  setCSR_Atomic_SEQ(ctrl_reg, offs_01, CTRL_REG + 1);
  setCSR_Atomic_SEQ(ctrl_reg, offs_02, CTRL_REG);

  printf("offs_3: %lx \n", offs_3);
  printf("offs_2: %lx \n", offs_2);
  printf("offs_1: %lx \n", offs_1);
  printf("offs_0: %lx \n", offs_0);

  printf("VADDR_WR_REG: %lx \n", &ctrl_reg[VADDR_WR_REG]);

  printf("offs_32: %x \n", offs_32);
  printf("offs_31: %x \n", offs_31);
  printf("CTRL_REG_2: %lx \n", &ctrl_reg[CTRL_REG_2]);
  printf("offs_22: %x \n", offs_22);
  printf("offs_21: %x \n", offs_21);
  printf("VADDR_RD_REG: %lx \n", &ctrl_reg[VADDR_RD_REG]);
  printf("offs_12: %x \n", offs_12);
  printf("offs_11: %x \n", offs_11);
  printf("CTRL_REG: %lx \n", &ctrl_reg[CTRL_REG]);
  printf("offs_02: %x \n", offs_02);
  printf("offs_01: %x \n", offs_01);
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

void compute_mi(coyote::cThread &coyote_thread, uint8_t *input_flt,
                uint8_t *input_ref, float *mutual_info, uint64_t *n_couples_mem,
                void *ctrl_reg) {
  uint32_t local_write_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE);
  std::cout << "Local write count: " << local_write_count << std::endl;
  uint32_t local_read_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ);

  size_t elems = DIMENSION * DIMENSION * n_couples_mem[0];
  uint32_t bytes = elems * sizeof(uint8_t);

  // Set Coyote thread arguments
  coyote::localSg sg_flt;

  memset(&sg_flt, 0, sizeof(coyote::localSg));
  sg_flt = {.addr = input_flt, .len = bytes, .dest = 0};

  int ctid = coyote_thread.getCtid();

  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_flt);

  /*
  int coper = 0;
  hipLaunchKernelGGL(gpu_invoke, dim3(1), dim3(1), 0, 0,
                     static_cast<volatile uint32_t *>(ctrl_reg), coper, bytes,
                     0, 0, 0, input_flt, 0, ctid);
  hipDeviceSynchronize();
                  */
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

  printf("Going to launch the kernel:\n");
  hipLaunchKernelGGL(launch_basic_test, dim3(1), dim3(1), 0, 0,
                     static_cast<volatile uint32_t *>(ctrl_reg));
  hipDeviceSynchronize();
  hipError_t val = hipGetLastError();
  printf("Value of the last error: %d \n", val);

  std::cout << "Control register set in Coyote thread" << std::endl;

  coyote::localSg sg_out;
  memset(&sg_out, 0, sizeof(coyote::localSg));
  sg_out = {.addr = mutual_info, .dest = 0, .len = sizeof(float)};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_WRITE, sg_out);

  while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) <=
         local_write_count)
    ;
  std::cout << "check completed: "
            << coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE)
            << std::endl;
  std::cout << "Mutual information computed in Coyote thread" << std::endl;
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
              << " <vfpga_id> <PET_folder> <CT_folder> <out_folder> "
                 "<tx> <ty> <ang> <runs> <gpu_id>\n";
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
  hipSetDevice(gpu_id);

  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, gpu_id);
  std::cout << "Device "
            << ": " << props.name << std::endl;

  int depth = 246; // depth of the volumes
  size_t elems = DIMENSION * DIMENSION * depth;
  uint32_t bytes = elems * sizeof(uint8_t);

  RigidWarpXYPlane hip_transform;
  std::cout << "Warming up HIP kernel...\n";

  for (int i = 0; i < 10; i++)
    hip_transform.run(0, 0, 0);

  std::cout << "Allocating memory for volumes...\n";
  uint8_t *float_cpu =
      (uint8_t *)malloc(DIMENSION * DIMENSION * depth * sizeof(uint8_t));
  // uint8_t* ref_cpu = (uint8_t*)malloc(DIMENSION * DIMENSION * depth *
  // sizeof(uint8_t));
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

  void *ctrl_reg = coyote_thread.get_ctrl_reg(gpu_id);

  // array of times for each run, to average
  std::vector<double> times(runs, 0.0);
  for (int i = 0; i < runs; i++) {

    std::cout << "Running HIP warp...\n";

    // measure time with precision
    std::chrono::high_resolution_clock::time_point reg_step_time_start =
        std::chrono::high_resolution_clock::now();
    // 2) Running Hip Warp
    hip_transform.moveToGPU(flt, float_cpu, DIMENSION, depth);

    double t =
        hip_transform.run_external(flt, out, tx, ty, ang, DIMENSION, depth);

    std::cout << "HIP exec time: " << t << " s\n";

    *n_couples_mem = static_cast<uint64_t>(depth);

    std::cout << "Computing Mutual Information...\n";

    auto time_start = std::chrono::high_resolution_clock::now();
    compute_mi(coyote_thread, out, ref, mutual_info, n_couples_mem, ctrl_reg);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;
    std::cout << "MI exec time: " << elapsed.count() << " s\n";

    auto reg_step_time_end = std::chrono::high_resolution_clock::now();
    elapsed = reg_step_time_end - reg_step_time_start;
    std::cout << "Registration step exec time: " << elapsed.count() << " s\n";
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

  free(float_cpu);

  save_volume_into_folder(float_cpu, DIMENSION, depth, out_dir);

  return 0;
}
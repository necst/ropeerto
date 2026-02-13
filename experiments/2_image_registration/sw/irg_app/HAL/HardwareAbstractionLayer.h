// HardwareAbstractionLayer.h
#pragma once

#include <any>
#include <string>

#include <hip/hip_runtime.h>

#include "../../HIPRigidWarp3D/src/hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hpp" // <-- proper header for your class
#include "images_io.h" // <-- header declaring read_volume_from_folder

#ifdef COYOTE_MODE
#include <utility>

// Coyote-specific includes
#include "cThread.hpp"

typedef struct {
  int device_index; // Device index for Coyote
  int vfpga_index;  // vFPGA index for Coyote
  int gpu_index; // GPU index
  bool p2p_mode;    // enable P2P buffers
} device; // Coyote mode uses int for device index and int for vFPGA index

#else

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

typedef struct {
  std::string xclbin_path; // Path to the xclbin file
  std::string kernel_name; // Name of the kernel in the xclbin
  int device_index;        // Device index for XRT
  // Note: XRT does not use vFPGA index, so we only have device_index
} device; // XRT mode uses a single int for device index

#endif

class HardwareAbstractionLayer {
public:
  /**
   * @param device   device struct representing the hardware device (either
   * Coyote or XRT)
   * @param resolution      Width and height of each slice (voxels)
   * @param depth           Number of slices in the volumefor Coyote)
   * @param transformer_    RigidWarpXYPlane helper for warping the volume
   */
  HardwareAbstractionLayer(const device &device, int resolution, int depth,
                           RigidWarpXYPlane &transformer_);
  ~HardwareAbstractionLayer();

  /// Load the reference volume from the given folder
  void load_ref(const std::string &folder);

  /// Load the filter volume from the given folder
  void load_flt(const std::string &folder);

  /// Run the FPGA kernel (if you still need it)
  float run_reg_step(float tx, float ty, float ang);

  /// Compute mutual information between the reference and transformed volume
  float compute_mi(uint8_t *curr_ptr_float);

  /// Warp a volume using a RigidWarpXYPlane helper
  void transform_volume(float tx, float ty, float ang, bool complete = true);

  /// Access the FPGA‐computed output buffer
  uint8_t *get_output() const { return ptr_out; }

  void printGPUCapabilities_HIP() {
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
      std::cerr << "Error in getting device count: " << hipGetErrorString(err)
                << std::endl;
      return;
    }

    for (int device = 0; device < deviceCount; device++) {
      hipDeviceProp_t props;
      err = hipGetDeviceProperties(&props, device);
      if (err != hipSuccess) {
        std::cerr << "Error in getting device properties: "
                  << hipGetErrorString(err) << std::endl;
        return;
      }

      std::cout << "\nGPU Device " << device << ": " << props.name << std::endl;
      std::cout << "Compute Capability: " << props.major << "." << props.minor
                << std::endl;
      std::cout << "Max threads per block: " << props.maxThreadsPerBlock
                << std::endl;
      std::cout << "Max threads in X-dimension: " << props.maxThreadsDim[0]
                << std::endl;
      std::cout << "Max threads in Y-dimension: " << props.maxThreadsDim[1]
                << std::endl;
      std::cout << "Max threads in Z-dimension: " << props.maxThreadsDim[2]
                << std::endl;
      std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
      std::cout << "Shared memory per block: " << props.sharedMemPerBlock
                << " bytes" << std::endl;
      std::cout << "Total global memory: " << props.totalGlobalMem << " bytes"
                << std::endl;
    }
  }

public:
#ifdef COYOTE_MODE
  // Coyote-specific members
  coyote::cThread coyote_thread;
  float *mutual_info;
  uint64_t *n_couples_mem;
  bool p2p_mode;
#else
  // XRT-specific members

  xrt::device device;
  xrt::kernel krnl;
  xrt::bo bo_ref, bo_flt, bo_out;
  xrt::run runner;

#endif

  uint8_t *ptr_ref = nullptr, *ptr_flt = nullptr, *ptr_out = nullptr,
          *float_cpu = nullptr;
  int resolution;
  int depth;
  RigidWarpXYPlane transformer;
  int counter = 0;
};

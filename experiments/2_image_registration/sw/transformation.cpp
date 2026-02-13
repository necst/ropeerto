#include <hip/hip_runtime.h>
#include <iostream>
// include folder for read volume from folder
#include "HIPRigidWarp3D/src/hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hpp"
#include "HIPRigidWarp3D/src/utils/images_io.h"

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

void transform_volume(RigidWarpXYPlane &transform, uint8_t *host_input_volume,
                      uint8_t *host_output_volume, int size, int depth,
                      float tx, float ty, float ang) {
  transform.transferToGPU(host_input_volume, size, depth);
  double exec_time = transform.run(tx, ty, ang);
  std::cout << "Execution time: " << exec_time << " seconds" << std::endl;
  transform.transferFromGPU(host_output_volume);
}

int main(int argc, char **argv) {
  const int size = 512;
  const int depth = 128;
  const int runs_warmup = 10;
  const int runs = 10;
  if (argc < 8) {
    std::cerr
        << "Usage: " << argv[0]
        << " <PET_folder> <CT_folder> <output_folder> <tx> <ty> <ang> <gpu_id>"
        << std::endl;
    return 1;
  }
  std::string PET_folder = argv[1];
  std::string CT_folder = argv[2];
  std::string output_folder = argv[3];
  std::string tx_str = argv[4];
  std::string ty_str = argv[5];
  std::string ang_str = argv[6];
  std::string gpu_id_str = argv[7];
  int tx = std::stoi(tx_str);
  int ty = std::stoi(ty_str);
  float ang = std::stof(ang_str);
  int gpu_id = std::stoi(gpu_id_str);
  hipSetDevice(gpu_id);
  printGPUCapabilities_HIP();
  std::cout << "Input volume size: " << size << "x" << size << "x" << depth
            << std::endl;
  std::cout << "Transformation parameters: tx = " << tx << ", ty = " << ty
            << ", ang = " << ang << std::endl;

  const size_t VOLUME = size * size * depth;
  const size_t VOLUME_BYTES = VOLUME * sizeof(uint8_t);

  uint8_t *host_input_volume = new uint8_t[VOLUME];
  uint8_t *host_output_volume = new uint8_t[VOLUME];
  uint8_t *host_reference_volume = new uint8_t[VOLUME];
  std::cout << "Loading input volume...\n";
  read_volume_from_folder(host_input_volume, size, depth, PET_folder);
  std::cout << "Loading reference volume...\n";
  read_volume_from_folder(host_reference_volume, size, depth, CT_folder);

  RigidWarpXYPlane transform; // Istanza della HAL per il kernel HIP

  for (int i = 0; i < runs_warmup; i++) {
    float warmup_tx = (float)rand() / RAND_MAX * 100.0f - 50.0f;
    float warmup_ty = (float)rand() / RAND_MAX * 100.0f - 50.0f;
    float warmup_ang = (float)rand() / RAND_MAX * 360.0f;
    transform.run(warmup_tx, warmup_ty, warmup_ang);
  }
  std::cout << "Warmup completed.\n";

  transform_volume(transform, host_input_volume, host_output_volume, size,
                   depth, tx, ty, ang);
  save_volume_into_folder(host_output_volume, size, depth, output_folder);

  return 0;
}

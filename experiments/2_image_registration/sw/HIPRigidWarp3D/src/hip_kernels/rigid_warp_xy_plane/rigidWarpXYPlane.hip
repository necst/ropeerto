#include "rigidWarpXYPlane.hpp"
#include <cmath> // Per cosf, sinf, sqrt, ceil
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

__device__ __host__ inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }


__global__ void rigidWarpXYPlane(const int size, const int depth,
                                 const uint8_t *input, uint8_t *output,
                                 const float translate_x,
                                 const float translate_y, const float ang) {

  const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

  const int total_pixels = size * size;
  const int total_elements = total_pixels * depth;

  const int threads_per_block = blockDim.x * blockDim.y;
  const int blocks_per_grid = gridDim.x * gridDim.y;
  const int threads_per_grid = threads_per_block * blocks_per_grid;
  const int elements_per_thread = ceilDiv(total_elements, threads_per_grid);

  const float half_size = size * 0.5f;
  const float p_cos = cosf(ang);
  const float p_sin = sinf(ang);

  for (int i = 0; i < elements_per_thread; i++) {
    const int global_idx =
        (block_idx * threads_per_block) + (i * threads_per_grid) + thread_idx;

    if (global_idx < total_elements) {
      const int pixel_idx = global_idx / depth;
      const int slice = global_idx % depth;
      const int row = pixel_idx / size;
      const int col = pixel_idx % size;

      if (row < size && col < size) {
        const float x_centered = col - half_size;
        const float y_centered = row - half_size;

        const float new_x =
            x_centered * p_cos - y_centered * p_sin + half_size - translate_x;
        const float new_y =
            x_centered * p_sin + y_centered * p_cos + half_size - translate_y;

        const int new_j = __float2int_rn(new_x);
        const int new_i = __float2int_rn(new_y);

        const bool out_of_bounds =
            new_i < 0 || new_i >= size || new_j < 0 || new_j >= size;

        if (out_of_bounds) {
          output[global_idx] = 0;
        } else {
          const int src_idx = (new_i * size + new_j) * depth + slice;
          output[global_idx] = input[src_idx];
        }
      }
    }
  }
}

#define HAL_PRINTF(fmt, ...) std::printf(" <HAL> " fmt, ##__VA_ARGS__)

RigidWarpXYPlane::RigidWarpXYPlane(const int device_id)
    : _device_id(device_id), device_input(nullptr), device_output(nullptr),
      _size(0), _depth(0) {}

void RigidWarpXYPlane::transferToGPU(const uint8_t *input, const int size,
                                     const int depth) {
  bool need_to_allocate = false;

  if (device_input == nullptr) {
    need_to_allocate = true;
  } else if (_size != size || _depth != depth) {
    need_to_allocate = true;

    hipFree(device_input);
    hipFree(device_output);
  }

  size_t bytes = size * size * depth * sizeof(uint8_t);

  if (need_to_allocate) {

    hipMalloc(&device_input, bytes);
    hipMalloc(&device_output, bytes);

    _size = size;
    _depth = depth;

    setupGrid(size); 
  }

  hipMemcpy(device_input, input, bytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();

}

void RigidWarpXYPlane::transferFromGPU(uint8_t *output) {
  size_t bytes = _size * _size * _depth * sizeof(uint8_t);
  hipMemcpy(output, device_output, bytes, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
}

void RigidWarpXYPlane::moveToGPU(uint8_t *dev_buffer,
                                 const uint8_t *host_buffer, const int size,
                                 const int depth) {
  size_t bytes = size * size * depth * sizeof(uint8_t);
  _size = size;
  _depth = depth;
  setupGrid(size);
  hipMemcpy(dev_buffer, host_buffer, bytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
}

void RigidWarpXYPlane::moveFromGPU(uint8_t *host_buffer,
                                   const uint8_t *dev_buffer, const int size,
                                   const int depth) {
  size_t bytes = size * size * depth * sizeof(uint8_t);
  hipMemcpy(host_buffer, dev_buffer, bytes, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
}

RigidWarpXYPlane::~RigidWarpXYPlane() {
  if (device_input != nullptr) {
    HAL_PRINTF("Freeing input buffer on the GPU\n");
    hipFree(device_input);
  }
  if (device_output != nullptr) {
    HAL_PRINTF("Freeing output buffer on the GPU\n");
    hipFree(device_output);
  }
}

void RigidWarpXYPlane::setupGrid(const dim3 blockSize, const dim3 gridSize) {
  _blockSize = blockSize;
  _gridSize = gridSize;

}

void RigidWarpXYPlane::setupGrid(const int threads_per_block) {
  dim3 blockSize(32, threads_per_block / 32);

  hipDeviceProp_t props;

  hipGetDeviceProperties(&props, _device_id);

  int max_blocks_per_sm = props.maxThreadsPerMultiProcessor / threads_per_block;
  int total_concurrent_blocks = props.multiProcessorCount * max_blocks_per_sm;

  int blocks_needed_x = static_cast<int>(std::ceil((float)_size / blockSize.x));
  int blocks_needed_y = static_cast<int>(std::ceil((float)_size / blockSize.y));
  int total_blocks_needed = blocks_needed_x * blocks_needed_y;

  int total_blocks = static_cast<int>(std::ceil((float)total_blocks_needed /
                                                total_concurrent_blocks)) *
                     total_concurrent_blocks;

  int grid_dim = static_cast<int>(std::ceil(std::sqrt(total_blocks)));
  dim3 gridSize(grid_dim, grid_dim);

  setupGrid(blockSize, gridSize);
}

double RigidWarpXYPlane::run(const float tx, const float ty, const float ang) {
  Timer timer;
  timer.start();

  hipLaunchKernelGGL(rigidWarpXYPlane, _gridSize, _blockSize, 0, 0, _size,
                     _depth, device_input, device_output, tx, ty, ang);
  hipDeviceSynchronize();

  double exec_time = timer.stop();
  return exec_time;
}

double RigidWarpXYPlane::run_external(const uint8_t *dev_input,
                                      uint8_t *dev_output, const float tx,
                                      const float ty, const float ang,
                                      uint32_t size, uint32_t depth) {
  _size = size;
  _depth = depth;
  Timer timer;
  timer.start();
  hipLaunchKernelGGL(rigidWarpXYPlane, _gridSize, _blockSize, 0, 0, _size,
                     _depth, dev_input, dev_output, tx, ty, ang);
  hipDeviceSynchronize();

  double exec_time = timer.stop();
  return exec_time;
}

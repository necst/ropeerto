#include "rigidWarpXYPlane.hpp"
#include <cmath> // Per cosf, sinf, sqrt, ceil
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

// Funzione inline per il calcolo efficiente di ceil(a/b)
__device__ __host__ inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }

/*
   Kernel HIP per effettuare la trasformazione rigida sul piano XY.
   La logica è identica alla versione CUDA.
*/
__global__ void rigidWarpXYPlane(const int size, const int depth,
                                 const uint8_t *input, uint8_t *output,
                                 const float translate_x,
                                 const float translate_y, const float ang) {
  // [1] Calcola l'indice del thread all'interno del blocco e il blocco
  // all'interno della grid
  const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

  // Calcola il numero totale di pixel ed elementi (pixel * depth)
  const int total_pixels = size * size;
  const int total_elements = total_pixels * depth;

  // [2] Calcola quanti elementi ogni thread deve processare, assicurando
  // accessi in memoria coalescati
  const int threads_per_block = blockDim.x * blockDim.y;
  const int blocks_per_grid = gridDim.x * gridDim.y;
  const int threads_per_grid = threads_per_block * blocks_per_grid;
  const int elements_per_thread = ceilDiv(total_elements, threads_per_grid);

  // [3] Precalcola alcuni valori utili
  const float half_size = size * 0.5f;
  const float p_cos = cosf(ang);
  const float p_sin = sinf(ang);

  // [4] Ciclo per processare il numero di elementi assegnati al thread
  for (int i = 0; i < elements_per_thread; i++) {
    const int global_idx =
        (block_idx * threads_per_block) + (i * threads_per_grid) + thread_idx;

    // [5] Verifica che l'indice globale sia nei limiti
    if (global_idx < total_elements) {
      // [6] Mappa l'indice globale in coordinate immagine e slice
      const int pixel_idx = global_idx / depth;
      const int slice = global_idx % depth;
      const int row = pixel_idx / size;
      const int col = pixel_idx % size;

      // [7] Controlla che la posizione XY sia valida
      if (row < size && col < size) {
        const float x_centered = col - half_size;
        const float y_centered = row - half_size;

        // [8] Applica la trasformazione rigida
        const float new_x =
            x_centered * p_cos - y_centered * p_sin + half_size - translate_x;
        const float new_y =
            x_centered * p_sin + y_centered * p_cos + half_size - translate_y;

        // [9] Interpolazione nearest neighbor
        const int new_j = __float2int_rn(new_x);
        const int new_i = __float2int_rn(new_y);

        // [10] Controlla se il nuovo pixel è fuori dai limiti
        const bool out_of_bounds =
            new_i < 0 || new_i >= size || new_j < 0 || new_j >= size;

        // [11] Scrive il risultato
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

// Macro per la stampa con prefisso "HAL"
#define HAL_PRINTF(fmt, ...) std::printf(" <HAL> " fmt, ##__VA_ARGS__)

// Costruttore della classe RigidWarpXYPlane
RigidWarpXYPlane::RigidWarpXYPlane(const int device_id)
    : _device_id(device_id), device_input(nullptr), device_output(nullptr),
      _size(0), _depth(0) {}

// Trasferisce il buffer di input dalla CPU alla GPU
void RigidWarpXYPlane::transferToGPU(const uint8_t *input, const int size,
                                     const int depth) {
  bool need_to_allocate = false;

  if (device_input == nullptr) {
    need_to_allocate = true;
    // HAL_PRINTF("Allocating input and output buffers on the GPU\n");
  } else if (_size != size || _depth != depth) {
    need_to_allocate = true;
    // HAL_PRINTF("(WARNING) input and output buffer sizes do not match the "
    //            "current size: Reallocating\n");

    hipFree(device_input);
    hipFree(device_output);
  }

  size_t bytes = size * size * depth * sizeof(uint8_t);

  if (need_to_allocate) {
    // HAL_PRINTF("Allocating %zu bytes\n", 2 * bytes);

    hipMalloc(&device_input, bytes);
    hipMalloc(&device_output, bytes);

    _size = size;
    _depth = depth;

    setupGrid(size); // Giuseppe: this value should be increased or making
                     // dependent on the device properties. 64 should be the
                     // warp size in AMD GPUs
  }

  //::cout << "Transfer size: " << bytes << " bytes\n";
  // measure time
  // std::chrono::high_resolution_clock::time_point start =
  //    std::chrono::high_resolution_clock::now();

  hipMemcpy(device_input, input, bytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "Transfer time: "
  //           << std::chrono::duration<double>(end - start).count()
  //           << " seconds\n";
}

// Trasferisce il buffer di output dalla GPU alla CPU
void RigidWarpXYPlane::transferFromGPU(uint8_t *output) {
  size_t bytes = _size * _size * _depth * sizeof(uint8_t);
  // std::cout << "Transfer size: " << bytes << " bytes\n";
  //  measure time
  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();
  hipMemcpy(output, device_output, bytes, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "Transfer time: "
  //           << std::chrono::duration<double>(end - start).count()
  //           << " seconds\n";
}

void RigidWarpXYPlane::moveToGPU(uint8_t *dev_buffer,
                                 const uint8_t *host_buffer, const int size,
                                 const int depth) {
  size_t bytes = size * size * depth * sizeof(uint8_t);
  _size = size;
  _depth = depth;
  setupGrid(size);
  // std::cout << "Transfer size: " << bytes << " bytes\n";
  //  measure time
  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();
  hipMemcpy(dev_buffer, host_buffer, bytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "Transfer time: "
  //           << std::chrono::duration<double>(end - start).count()
  //           << " seconds\n";
}

void RigidWarpXYPlane::moveFromGPU(uint8_t *host_buffer,
                                   const uint8_t *dev_buffer, const int size,
                                   const int depth) {
  size_t bytes = size * size * depth * sizeof(uint8_t);
  // std::cout << "Transfer size: " << bytes << " bytes\n";
  //  measure time
  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();
  hipMemcpy(host_buffer, dev_buffer, bytes, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "Transfer time: "
  //           << std::chrono::duration<double>(end - start).count()
  //           << " seconds\n";
}

// Distruttore della classe
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

// Imposta le dimensioni della grid e del blocco per il kernel HIP
void RigidWarpXYPlane::setupGrid(const dim3 blockSize, const dim3 gridSize) {
  _blockSize = blockSize;
  _gridSize = gridSize;

  /*
  HAL_PRINTF("Setting up grid and block sizes\n");
  HAL_PRINTF(" - grid size: %d x %d\n", gridSize.x, gridSize.y);
  HAL_PRINTF(" - block size: %d x %d\n", blockSize.x, blockSize.y);
  */
}

// Imposta la grid in base al numero di threads per blocco desiderato
void RigidWarpXYPlane::setupGrid(const int threads_per_block) {
  dim3 blockSize(32, threads_per_block / 32);

  hipDeviceProp_t props;

  hipGetDeviceProperties(&props, _device_id);

  /*
  // print device props
  std::cout << "Device Properties:\n";
  std::cout << " - Name: " << props.name << "\n";
  std::cout << " - Total Global Memory: "
            << props.totalGlobalMem / (1024 * 1024) << " MB\n";
  std::cout << " - Shared Memory per Block: "
            << props.sharedMemPerBlock / (1024) << " KB\n";
  std::cout << " - Registers per Block: " << props.regsPerBlock << "\n";
  std::cout << " - Warp Size: " << props.warpSize << "\n";
  std::cout << " - Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
  std::cout << " - Max Threads per MultiProcessor: "
            << props.maxThreadsPerMultiProcessor << "\n";
  std::cout << " - Max Blocks per MultiProcessor: "
            << props.maxBlocksPerMultiProcessor << "\n";
 */
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

// Esegue il kernel HIP e restituisce il tempo di esecuzione in secondi
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
  // std::cout << "Running external kernel with size: " << _size
  //           << ", depth: " << _depth << "\n";
  Timer timer;
  timer.start();
  hipLaunchKernelGGL(rigidWarpXYPlane, _gridSize, _blockSize, 0, 0, _size,
                     _depth, dev_input, dev_output, tx, ty, ang);
  // rigidWarpXYPlane<<<_gridSize, _blockSize, 0, stream>>>(
  //  _size, _depth, dev_input, dev_output, tx, ty, ang);
  // rigidWarpXYPlane<<<_blockSize, 0>>>(_size, _depth, dev_input,
  // dev_output, tx,
  //                                    ty, ang);
  // size_t bytes = size * size * depth *
  // sizeof(uint8_t);

  // hsa_memory_copy(dev_output, dev_input,
  // bytes);
  // hipStreamSynchronize(stream);
  hipDeviceSynchronize();

  double exec_time = timer.stop();
  return exec_time;
}

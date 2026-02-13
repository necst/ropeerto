
#include "rigidWarpXYPlane.cuh"

#include <cstdio>
#include <device_launch_parameters.h>

#include <timer.hpp>

// Efficient way of calculating ceil(a/b) without using floating point arithmetic
__device__ __host__ inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

/*
see thread hierarchu / thread indexing in CUDA:
    https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

How it works:
 - [1] Identify thread and block indexes within the grid.
 - [2] Compute how many pixels this thread will compute 
        (we need to iterate over such number of pixels).
 - [3] Precompute some values.
 - [4] Compute the "global index" of the thread 
        (thus, also considering block's index in the grid).
 - [5] Check if the global index is within the bounds of the total number of elements
        (if not, we don't need to do anything and this thread has finished working).
 - [6] Compute the pixel index depth-wise: 
        - consider Tidx as the current thread index
        - the thread "on the left" (a.k.a. the preeceding thread) will have index Tidx-1
        - threads Tidx and Tidx-1 will likely(?) process consecutive memory locations 
            (coalesced access; they are within the same warp)
        - thus, threads Tidx-1 and Tidx should work on the same XY location, but subsequent slices S and S+1
        - since by moving to the next thread the Tidx increases by one, the slice index should increase by one as well
            (slice_idx = global_idx % depth)
        - while the XY location is the same
            (pixel_idx = global_idx / depth)
        - this ends the depth-wise thread->pixel mapping.
        - finally, we compute the row and col coordinates of the pixel (in output space, in XY plane).
 - [7] like point 5
 - [8] Apply the transformation
 - [9] Apply nearest neighbor interpolation
 - [10] What the fetcher usually does
        - if the pixel to read (input space) is within the bounds of the input image, the read it
        - otherwise, set the output pixel to 0
*/

__global__
void rigidWarpXYPlane(
    const int size, const int depth, 
    const uint8_t *input, uint8_t *output, 
    const float translate_x, const float translate_y, const float ang) 
{
    // <!> Reorganize thread indexing to process consecutive memory locations within a warp.
    //     This will ensure coalesced memory access by mapping consecutive threads to consecutive memory locations.
    //     Basically, depth-wise thread->pixel mapping.

    // [1]
    const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Calculate total number of elements to process
    const int total_pixels = size * size;
    const int total_elements = total_pixels * depth;
    
    // Process elements in a way that ensures coalesced access
    const int threads_per_block = blockDim.x * blockDim.y;
    const int blocks_per_grid = gridDim.x * gridDim.y;
    const int threads_per_grid = threads_per_block * blocks_per_grid;
    // [2]
    const int elements_per_thread = ceilDiv(total_elements, threads_per_grid);

    // [3]
    const float half_size = size * 0.5f;
    const float p_cos = cosf(ang);
    const float p_sin = sinf(ang);
    
    // [3]
    for (int i = 0; i < elements_per_thread; i++) {
        // [4]
        const int global_idx = (block_idx * threads_per_block) + (i * threads_per_grid) + thread_idx;
        
        // [5]
        if (global_idx < total_elements) {
            // [6]
            // Convert linear index back to image coordinates (depth-wise thread->pixel mapping)
            const int pixel_idx = global_idx / depth;
            const int slice = global_idx % depth;
            const int row = pixel_idx / size;
            const int col = pixel_idx % size;
            
            // [7]
            if (row < size && col < size) {
                const float x_centered = col - half_size;
                const float y_centered = row - half_size;
                
                // [8]
                // TODO use intrinsic fmaf
                const float new_x = x_centered * p_cos - y_centered * p_sin + half_size - translate_x;
                const float new_y = x_centered * p_sin + y_centered * p_cos + half_size - translate_y;
                // const float new_x = fmaf(x_centered, p_cos, -y_centered * p_sin + half_size - translate_x);
                // const float new_y = fmaf(x_centered, p_sin, y_centered * p_cos + half_size - translate_y);
                
                // [9]
                const int new_j = static_cast<int>(roundf(new_x));;
                const int new_i = static_cast<int>(roundf(new_y));
                
                // [10]
                const bool out_of_bounds = new_i < 0 || new_i >= size || new_j < 0 || new_j >= size;
                
                // [11]
                if (out_of_bounds) { // TODO use predication
                    output[global_idx] = 0;
                } else {
                    // depth-wise data-layout
                    const int src_idx = (new_i * size + new_j) * depth + slice;
                    output[global_idx] = input[src_idx];

                    // also read reference
                    // update joint histogram
                    // update horizontal and vertical histograms
                }
            }
        }
    }
}


// define HAL_PRINTF which calls printf with prefix "HAL: "
#define HAL_PRINTF(fmt, ...) std::printf(" <HAL> " fmt, ##__VA_ARGS__)


/**
 * @brief Construct a new RigidWarpXYPlane::RigidWarpXYPlane object
 * 
 * @param device_id the ID of the GPU device to use
*/
RigidWarpXYPlane::RigidWarpXYPlane(const int device_id)  
    : _device_id(device_id) { }

/**
 * @brief Transfer the input buffer from the host to the GPU
 * 
 * @param input the input buffer on the host
 * @param size the size of the input buffer in the x and y dimensions
 * @param depth the depth of the input buffer in the z dimension
 * 
*/
void RigidWarpXYPlane::transferToGPU(const uint8_t* input, const int size, const int depth) {
    // if the pointer is null or _size or _depth is changed, then reallocate memory
    bool need_to_allocate = false;

    if (device_input == nullptr) {
        need_to_allocate = true;
        HAL_PRINTF("Allocating input and output buffers on the GPU\n");
    } else if (_size != size || _depth != depth) {
        need_to_allocate = true;
        HAL_PRINTF("(WARNING) input and output buffer sizes do not match the current size: Reallocating\n");

        // Free the memory
        cudaFree(device_input);
        cudaFree(device_output);
    }

    size_t bytes = size * size * depth * sizeof(uint8_t);

    if (need_to_allocate) {
        HAL_PRINTF("Allocating %zu bytes\n", 2*bytes);

        // Allocate memory on the GPU
        cudaMalloc(&device_input, bytes);
        cudaMalloc(&device_output, bytes);

        _size = size;
        _depth = depth;

        setupGrid(size);
    }

    // Copy the input buffer from the CPU to the GPU
    cudaMemcpy(device_input, input, bytes, cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer the output buffer from the GPU to the host
 * 
 * @param output the output buffer on the host
*/
void RigidWarpXYPlane::transferFromGPU(uint8_t* output) {
    size_t bytes = _size * _size * _depth * sizeof(uint8_t);

    // Copy the output buffer from the GPU to the CPU
    cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost);
}

RigidWarpXYPlane::~RigidWarpXYPlane() {
    if (device_input != nullptr) {
        HAL_PRINTF("Freeing input buffer on the GPU\n");
        cudaFree(device_input);
    }

    if (device_output != nullptr) {
        HAL_PRINTF("Freeing output buffer on the GPU\n");
        cudaFree(device_output);
    }
}

/**
 * @brief Set the grid and block sizes for the CUDA kernel
 * 
 * @param blockSize the block size
 * @param gridSize the grid size
*/
void RigidWarpXYPlane::setupGrid(const dim3 blockSize, const dim3 gridSize) {
    _blockSize = blockSize;
    _gridSize = gridSize;

    HAL_PRINTF("Setting up grid and block sizes\n");
    HAL_PRINTF(" - grid size: %d x %d\n", gridSize.x, gridSize.y);
    HAL_PRINTF(" - block size: %d x %d\n", blockSize.x, blockSize.y);
}

/**
 * @brief Set up the grid and block sizes for the CUDA kernel
 * 
 * @param threads_per_block the number of threads per block
*/
void RigidWarpXYPlane::setupGrid(const int threads_per_block) {
    // The workload is divided into blocks of threads_per_block threads.
    // To maximize parallelism, we want to maximize the number of blocks.
    // At the same time, we want to maximize the number of threads per block that is a multiple of 32,
    //  because the number of threads that can run concurrently in a warp is 32.
    // Also, if the 32 threads in a warp access consecutive memory locations,
    //  then the memory access is coalesced, which is more efficient.

    // This way of computing the block size might be completely wrong.

    dim3 blockSize(32, threads_per_block / 32);

    // Get the device properties
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, _device_id);

    // Calculate blocks per SM (Streaming Multiprocessor)
    int max_blocks_per_sm = props.maxThreadsPerMultiProcessor / threads_per_block;

    // Total concurrent blocks possible on the GPU
    int total_concurrent_blocks = props.multiProcessorCount * max_blocks_per_sm;

    // Calculate needed grid size based on image dimensions
    int blocks_needed_x = ceil((float)_size / blockSize.x);
    int blocks_needed_y = ceil((float)_size / blockSize.y);
    int total_blocks_needed = blocks_needed_x * blocks_needed_y;

    // Round up to the next multiple of total_concurrent_blocks
    int total_blocks = ceil((float)total_blocks_needed / total_concurrent_blocks) * total_concurrent_blocks;
    
    // Calculate grid dimensions that are as square as possible
    int grid_dim = ceil(sqrt(total_blocks));
    dim3 gridSize(grid_dim, grid_dim);

    setupGrid(blockSize, gridSize);
}

/**
 * @brief Run the CUDA kernel
 * 
 * @param tx the translation in the x direction
 * @param ty the translation in the y direction
 * @param ang the rotation angle
 * 
 * @return the execution time in seconds
*/
double RigidWarpXYPlane::run(const float tx, const float ty, const float ang) {
    Timer timer;
    timer.start();

    rigidWarpXYPlane<<<_gridSize, _blockSize>>>(_size, _depth, device_input, device_output, tx, ty, ang);
    cudaDeviceSynchronize();

    double exec_time = timer.stop();

    return exec_time;
}

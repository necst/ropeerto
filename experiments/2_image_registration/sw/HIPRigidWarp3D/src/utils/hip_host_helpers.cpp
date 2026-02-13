#include "hip_host_helpers.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

void printGPUCapabilities() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, device);
        
        std::printf("\nGPU Device %d: %s\n", device, props.name);
        std::printf("Compute Capability: %d.%d\n", props.major, props.minor);
        std::printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
        std::printf("Max threads in X-dimension: %d\n", props.maxThreadsDim[0]);
        std::printf("Max threads in Y-dimension: %d\n", props.maxThreadsDim[1]);
        std::printf("Max threads in Z-dimension: %d\n", props.maxThreadsDim[2]);
        std::printf("Number of SMs: %d\n", props.multiProcessorCount);
        std::printf("Shared memory per block: %zu bytes\n", props.sharedMemPerBlock);
        std::printf("Total global memory: %zu bytes\n\n", props.totalGlobalMem);
    }
}


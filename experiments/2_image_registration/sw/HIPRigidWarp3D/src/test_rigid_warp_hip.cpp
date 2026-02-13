#include "tests.h"

#include <hip/hip_runtime.h>
#include <iostream>

#include "hip_kernels/rigid_warp_xy_plane/rigidWarpXYPlane.hpp"

#include <images_io.h>
#include <hip_host_helpers.h>   
#include <args_parser.h>

void printGPUCapabilities_HIP() {
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

int test_rigid_warp_hip(int argc, char** argv) {

    // Set Device 0 with Hip and check correctness
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        printf("Errore nel recuperare il numero di dispositivi: %s\n", hipGetErrorString(err));
        return -1;
    }
    
    int device_id = 1;
    err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        printf("Errore nell'impostare il device %d: %s\n", device_id, hipGetErrorString(err));
        return -1;
    }

    rigid_warp_parsed_args args = rigid_warp_parse_args(argc, argv); 

    std::printf("Size:  %4d\n", args.size);
    std::printf("Depth: %4d\n", args.depth);
    std::printf("tx:  %12f\n", args.tx);
    std::printf("ty:  %12f\n", args.ty);
    std::printf("ang: %12f\n", args.ang);

    printGPUCapabilities_HIP(); // Assicurarsi che questa funzione usi le API HIP

    const size_t VOLUME = args.size * args.size * args.depth;
    const size_t VOLUME_BYTES = VOLUME * sizeof(uint8_t);

    uint8_t *host_input_volume = new uint8_t[VOLUME];
    uint8_t *host_output_volume = new uint8_t[VOLUME];
    
    std::printf("Load volume...\n");
    //generate_example_image(host_input_volume, args.size, args.depth);
    // esempio di lettura di un volume da cartella:
    read_volume_from_folder(host_input_volume, args.size, args.depth, "data/input/PET");
    
    std::string input_folder = "data/input/generated";
    std::printf("Saving input image into folder: %s\n", input_folder.c_str());
    // save_volume_into_folder(host_input_volume, args.size, args.depth, input_folder);

    // ----------------------------------------

    RigidWarpXYPlane transform; // Istanza della HAL per il kernel HIP

    std::printf("Loading volume into GPU...\n");
    transform.transferToGPU(host_input_volume, args.size, args.depth);

    std::printf("Running rigidWarpXYPlane %d time%s for warmup...\n", args.runs_warmup, args.runs_warmup != 1 ? "s" : "");

    // Esecuzione di run di warmup con trasformazioni casuali
    for (int i = 0; i < args.runs_warmup; i++) {
        float warmup_tx = (float)rand() / RAND_MAX * 100.0f - 50.0f;
        float warmup_ty = (float)rand() / RAND_MAX * 100.0f - 50.0f;
        float warmup_ang = (float)rand() / RAND_MAX * 360.0f;
        transform.run(warmup_tx, warmup_ty, warmup_ang);
    }

    std::printf("Running rigidWarpXYPlane %d time%s...\n", args.runs, args.runs != 1 ? "s" : "");
    
    double min_exec_time = std::numeric_limits<double>::max();
    double max_exec_time = std::numeric_limits<double>::min();
    double mean_exec_time = 0.0;

    for (int i = 0; i < args.runs; i++) {
        std::printf("Run %3d", i + 1);
        double exec_time = transform.run(args.tx, args.ty, args.ang);
        std::printf(" [%f s]\n", exec_time);

        min_exec_time = std::min(min_exec_time, exec_time);
        max_exec_time = std::max(max_exec_time, exec_time);
        mean_exec_time += exec_time;
    }
    mean_exec_time /= args.runs;

    std::printf("\nExecution times:\n");
    std::printf("Min: %10f s\n", min_exec_time);
    std::printf("Max: %10f s\n", max_exec_time);
    std::printf("Avg: %10f s\n\n", mean_exec_time);

    std::printf("Loading volume from GPU...\n");
    transform.transferFromGPU(host_output_volume);

    std::string output_folder = "data/output/hip_transformed_volume";
    std::printf("Saving output image into folder: %s\n", output_folder.c_str());
    save_volume_into_folder(host_output_volume, args.size, args.depth, output_folder);

    return 0;
}

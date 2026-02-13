/**
 * This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
 *
 * MIT Licence
 * Copyright (c) 2025-2026, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Standard includes
#include <ctime>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <utility>
#include <iostream>
#include <filesystem>

// AMD GPU management & run-time libraries
#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>

// External library for easier parsing of CLI arguments by the executable
#include <boost/program_options.hpp>

// Coyote-specific includes
#include <coyote/cBench.hpp>
#include <coyote/cThread.hpp>

// Current bitstream is only synthesized with one vFPGA for simple pass-through data movement
#define DEFAULT_VFPGA_ID 0

// Typedefs
// In HIP, "standard" (malloc, memalign) CPU memory may not be optimal for GPU => CPU transfers;
// hence, it's recommended to use pinned memory (hipHostMalloc); however, such pinned memory
// uses regular pages and causes many TLB misses in Coyote and lowers CPU => FPGA performance.
// The best approach is to use Coyote's hugepage memory allocator and then register it with HIP
// However, when registered with HIP, one must keep track of two pointers: one from the GPU
// point of view and one from the CPU point of view; hence, we use a pair to keep track of the CPU memory pointers
// See the variable mem_alloc_type below and the following link:
// https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html#gab8258f051e1a1f7385f794a15300e674
typedef std::pair<void*, void*> cpu_mem_pair_t;

// Utility macro to check HIP errors; usage: HIP_CHECK(hipFunction(...));
#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

/**
 * @brief Class for monitoring GPU performance
 *
 * This class uses the ROCm SMI library to monitor the GPU power and utilization.
 * It runs in a separate thread, sampling the GPU power and utilization at regular intervals.
 * The results can be accessed via avg_power() and avg_util() methods.
 */
class PerfMonitor {
public:
    PerfMonitor(uint32_t gpu_id) : gpu_id(gpu_id), running(false) {
        rsmi_init(0);
    }

    ~PerfMonitor() {
        stop();
        rsmi_shut_down();
    }

    void start() {
        running = true;
        samples.clear();
        sampling_thread = std::thread([&]() {
            while (running) {
                uint64_t power;
                RSMI_POWER_TYPE power_type = RSMI_CURRENT_POWER;
                rsmi_dev_power_get(gpu_id, &power, &power_type);    // GPU power, Microwatts

                uint32_t utilization;
                rsmi_dev_busy_percent_get(gpu_id, &utilization);    // GPU utilization in %

                samples.push_back({(double) power / 1e6, (double) utilization});

                // Sleep for 100 microseconds before the next sample
                // Hence, this sampling only makes sense for larger transfers
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    void stop() {
        running = false;
        if (sampling_thread.joinable()) sampling_thread.join();
    }

    double avg_power() const {
        double sum = 0; 
        for (auto &s : samples) {
            sum += s.first;
        }
        return samples.empty() ? 0.0 : sum / samples.size();
    }

    double avg_util() const {
        double sum = 0; 
        for (auto &s : samples) {
            sum += s.second;
        }
        return samples.empty() ? 0.0 : sum / samples.size();
    }

private:
    bool running;
    uint32_t gpu_id;
    std::thread sampling_thread;
    std::vector<std::pair<double, double>> samples; // (GPU power W, utilization %)
};

/**
 * @brief Utility class for collecting and storing performance metrics
 *
 * This class collects performance metrics such as latencies, GPU power, and GPU utilization.
 * It provides methods to add new metrics, retrieve all metrics of a specific kind, and calculate
 */
class PerfMetrics {    
public:
    PerfMetrics() {};

    void add_new(double latency, double gpu_power, double gpu_util) {
        latencies.push_back(latency);
        gpu_powers.push_back(gpu_power);
        gpu_utils.push_back(gpu_util);
    }

    std::vector<double> get_all(std::string kind) {
        if (kind == "latency") {
            return latencies;
        } else if (kind == "gpu_power") {
            return gpu_powers;
        } else if (kind == "gpu_util") {
            return gpu_utils;
        } else {
            throw std::runtime_error("Unknown kind of performance metric requested: " + kind);
        }
    }

    double get_average(std::string kind) {
        if (kind == "latency") {
            return vector_average(latencies);
        } else if (kind == "gpu_power") {
            return vector_average(gpu_powers);
        } else if (kind == "gpu_util") {
            return vector_average(gpu_utils);
        } else {
            throw std::runtime_error("Unknown kind of performance metric requested: " + kind);
        }
    }

private:
    std::vector<double> latencies;
    std::vector<double> gpu_powers;
    std::vector<double> gpu_utils;

    double vector_average(std::vector<double> &vec) {
        double sum = 0.0;
        for (const auto &val : vec) {
            sum += val;
        }
        return vec.empty() ? 0.0 : sum / vec.size();
    }
};

/**
 * @brief Utility function which stores the results of a benchmark to a CSV file
 *
 * @param file_name Name of the CSV file to store the results in (will be created if it doesn't exist, and appended to if it does exist)
 * @param mode 1 for P2P, 0 for non-P2P
 * @param n_runs Number of times the test was repeated
 * @param size Size of the transfer in bytes
 * @param transfers Number of parallel transfers to launch
 * @param avg_latency Average time taken for the transfers in us
 * @param avg_throughput Average throughput of the transfers in GB/s
 * @param avg_gpu_power Average GPU power during the transfers in W
 * @param avg_gpu_util Average GPU utilization during the transfers in %xw
 */
void log_to_file(
    std::string file_name, bool mode, unsigned int n_runs, unsigned int size, unsigned int transfers,  
    double avg_latency, double avg_throughput, double avg_gpu_power, double avg_gpu_util
) {
    // Get the executable path
    std::filesystem::path exec_path = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path results_dir = exec_path.parent_path() / ".." / ".." / ".." / "results";
    
    // Create results directory if it doesn't exist
    std::filesystem::create_directories(results_dir);
    
    // Full path to the results file
    std::filesystem::path file_path = results_dir / file_name;
    
    // Check if file exists to determine if we need to write the header
    bool file_exists = std::filesystem::exists(file_path);
    
    // Open file in append mode
    std::ofstream outfile(file_path, std::ios::app);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open results file: " + file_path.string());
    }
    
    // Write CSV header if the file is new
    if (!file_exists) {
        outfile << "timestamp,mode,n_runs,size,transfers,avg_latency,avg_throughput,avg_gpu_power,avg_gpu_util\n";
    }
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&now_time, &tm_buf);
    
    // Write data row with proper CSV formatting
    outfile << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << ","
            << mode << ","
            << n_runs << ","
            << size << ","
            << transfers << ","
            << std::fixed << std::setprecision(6) << avg_latency << ","
            << avg_throughput << ","
            << avg_gpu_power << ","
            << avg_gpu_util << "\n";
    
    outfile.close();
}

/**
 * @brief Utility function which runs the benchmark
 *
 * This function runs multiple iterations of the benchmark, measuring the time taken for transfers.
 * First, it set the src (based on inputs) and dst (to zero) buffers on the GPU.
 * Then, before every iteration of the benchmark, it clears the completion counters and synchronizes the GPU.
 * Finally, this function defines the benchmark function, which performs the transfers and measures the time taken.
 *
 * @param coyote_thread Coyote thread to use for the transfers
 * @param hip_streams_d2h HIP streams for GPU to CPU transfers in non-P2P mode
 * @param hip_streams_h2d HIP streams for CPU to GPU transfers in non-P2P mode
 * @param gpu_src GPU source buffer (P2P or non-P2P mode)
 * @param gpu_dst GPU destination buffer (P2P or non-P2P mode)
 * @param cpu_src CPU source buffer (only in non-P2P mode, intermediate buffer between GPU and FPGA)
 * @param cpu_dst CPU destination buffer (only in non-P2P mode, intermediate buffer between FPGA and GPU)    
 * @param inputs Input data buffer (used to set the source data for the transfers, for validation purposes)
 * @param results Results buffer (used to copy the results back from the GPU, for validation purposes)
 * @param size Size of the transfer in bytes
 * @param transfers Number of parallel transfers to launch
 * @param n_runs Number of times to repeat the test
 * @param mode 1 for P2P, 0 for non-P2P
 * @param gpu_perf_monitoring Whether GPU performance monitoring is enabled or not (if not, the perf_monitor can be ignored)
 * @param perf_monitor Performance monitor to sample GPU power and utilization during the benchmark
 * @return PerfMetrics, which contains the average time taken for the transfers, GPU power, and GPU utilization
 */
PerfMetrics run_bench(
    coyote::cThread &coyote_thread, std::vector<hipStream_t> &hip_streams_d2h, std::vector<hipStream_t> &hip_streams_h2d,
    int *gpu_src, int *gpu_dst, cpu_mem_pair_t cpu_src, cpu_mem_pair_t cpu_dst, int* inputs, int* results,
    unsigned int size, unsigned int transfers, unsigned int n_runs, bool mode, bool gpu_perf_monitoring, PerfMonitor &perf_monitor
) {
    // Initialize metrics to return to the user
    PerfMetrics perf_metrics;

    // Indicates whether we are doing warm-up runs currently or not
    bool warm_up;
    
    // Randomly set the source data between -512 and +512; initialise destination memory to 0
    for (int i = 0; i < size / sizeof(int); i++) {
        inputs[i] = rand() % 1024 - 512;     
        results[i] = 0;         
    }    
    
    // Copy the initiated inputs/results to the corresponding GPU buffers
    HIP_CHECK(hipMemcpy(gpu_src, inputs, size, hipMemcpyHostToDevice)); 
    HIP_CHECK(hipMemcpy(gpu_dst, results, size, hipMemcpyHostToDevice)); 

    // For non-P2P, to keep track what streams have completed
    bool gpu_to_cpu_done;
    std::vector<bool> stream_completed;
    for (int i = 0 ; i < transfers; i++) {
        stream_completed.push_back(false);
    }

    // Set up Coyote scatter-gather (SG) entry
    coyote::localSg src_sg, dst_sg;
    if (mode) {
        src_sg = { .addr = gpu_src, .len = size };
        dst_sg = { .addr = gpu_dst, .len = size };
    } else {
        src_sg = { .addr = cpu_src.second, .len = size }; // Point to CPU memory, from CPU's point of view
        dst_sg = { .addr = cpu_dst.second, .len = size }; // Point to CPU memory, from CPU's point of view
    }

    // Function called before every iteration of the benchmark, can be used to clear previous flags, states etc.
    auto prep_fn = [&]() {
        // Clear the completion counters, so that the test can be repeated multiple times independently
        // Essentially, sets the result from the function checkCompleted(...) to zero
        coyote_thread.clearCompleted();

        // Also, synchronize the GPU
        HIP_CHECK(hipDeviceSynchronize());

        // Reset stream completions
        for (int i = 0 ; i < transfers; i++) {
            stream_completed[i] = false;
        }

        // Wait for 1 second if measuring GPU power/utilization, for the metrics to stabilize (as mentioned in the paper)
        if (gpu_perf_monitoring) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    };

    // Execute benchmark
    auto bench_fn = [&]() {
        // Start performance monitor and take timestamp
        if (gpu_perf_monitoring) {
            perf_monitor.start();
        }
        auto begin_time = std::chrono::high_resolution_clock::now();
        
        // Non-P2P case
        if (!mode) {
            // First, do a memcpy from GPU to to CPU
            // For throughput tests, launch multiple transfers in parallel; for latency tests, launch one
            // NOTE: No error checking here, not to artificially increase the latency
            int ret_val;
            for (int i = 0; i < transfers; i++) {
                ret_val = hipMemcpyAsync(cpu_src.first, gpu_src, size, hipMemcpyDeviceToHost, hip_streams_d2h[i]);
            }

            // As soon as one is finished, launch its corresponding Coyote transfer: CPU => vFPGA => CPU (non-P2P)
            gpu_to_cpu_done = false;
            while (!gpu_to_cpu_done) {
                gpu_to_cpu_done = true;
                for (unsigned int i = 0; i < transfers; i++) {
                    if (hipStreamQuery(hip_streams_d2h[i]) != hipSuccess) {
                        gpu_to_cpu_done = false;
                    } else {
                        if (!stream_completed[i]) {
                            coyote_thread.invoke(coyote::CoyoteOper::LOCAL_TRANSFER, src_sg, dst_sg);
                            stream_completed[i] = true;
                        }
                    }
                }        
            }

            // Now, as soon as one Coyote transfer is finished, launch its corresponding GPU transfer: CPU => GPU
            unsigned int completed_coyote = 0;
            while (completed_coyote < transfers) {
                unsigned int old_completed_coyote = completed_coyote;
                completed_coyote = coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER);

                for (unsigned int i = old_completed_coyote; i < completed_coyote; i++) {
                    // Launch the GPU transfer for the completed Coyote transfer
                    // NOTE: No error checking here, not to artificially increase the latency
                    ret_val = hipMemcpyAsync(gpu_dst, cpu_dst.first, size, hipMemcpyHostToDevice, hip_streams_h2d[i]);
                }
            
            }

            // Simply synchronize the device to ensure that all transfers are complete
            ret_val = hipDeviceSynchronize();

        // P2P case
        } else {
            // For P2P, do a GPU => vFPGA => GPU transfer
            for (int i = 0; i < transfers; i++) {
                coyote_thread.invoke(coyote::CoyoteOper::LOCAL_TRANSFER, src_sg, dst_sg);
            }

            // Wait until all transfers are complete
            while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER) != transfers) {}
        
            // Synchronize just to make sure consistency with P2P case (though not needed)
            int ret_val = hipDeviceSynchronize();
        }

        // Stop performance monitor and store results, if not warm-up
        if (gpu_perf_monitoring) {
            perf_monitor.stop();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
        if (!warm_up) {
            perf_metrics.add_new(
                elapsed_time, perf_monitor.avg_power(), perf_monitor.avg_util()
            );
        }
    };


    // Warm-up runs
    warm_up = true;
    for(int j = 0; j < 50; j++) {
        prep_fn();
        bench_fn();
    }

    // Run benchmark
    warm_up = false;
    std::vector<double> times;
    for(int j = 0; j < n_runs; j++) {
        prep_fn();
        bench_fn();
    }

    // Make sure destination matches the source + 1 (the vFPGA logic in perf_local adds 1 to every 32-bit element, i.e. integer)
    HIP_CHECK(hipMemcpy(results, gpu_dst, size, hipMemcpyDeviceToHost)); 
    for (int i = 0; i < size / sizeof(int); i++) {
        if ((inputs[i] + 1) != results[i]) {
            throw std::runtime_error("Wrong result!");
        }
    }

    // Return average time taken for the data transfer
    return perf_metrics;
}

int main(int argc, char *argv[])  {
    // CLI arguments
    // P2P = 1, non-P2P = 0
    bool mode;
    
    // Target GPU
    unsigned int gpu_id;

    // Samples GPU power & utilization if set§
    bool gpu_perf_monitoring;

    // Standard benchmark parameters
    unsigned int min_size, max_size, n_runs, n_transfers;
    
    // Output file for logging results
    std::string output_file;
    
    boost::program_options::options_description runtime_options("Coyote Perf GPU Options");
    runtime_options.add_options()
        ("gpu_id,g", boost::program_options::value<unsigned int>(&gpu_id)->default_value(0), "Target GPU")
        ("mode,m", boost::program_options::value<bool>(&mode)->default_value(true), "Benchmark mode: 1 (P2P) or 0 (non-P2P, baseline)")
        ("gpu_perf_monitoring,p", boost::program_options::value<bool>(&gpu_perf_monitoring)->default_value(false), "Sample GPU power and utilization during the benchmark")
        ("runs,r", boost::program_options::value<unsigned int>(&n_runs)->default_value(50), "Number of times to repeat the test")
        ("transfers,t", boost::program_options::value<unsigned int>(&n_transfers)->default_value(1), "Number of parallel transfers to launch")
        ("min_size,x", boost::program_options::value<unsigned int>(&min_size)->default_value(256), "Starting (minimum) transfer size [B]")
        ("max_size,X", boost::program_options::value<unsigned int>(&max_size)->default_value(2 * 1024 * 1024), "Ending (maximum) transfer size [B]")
        ("output_file,f", boost::program_options::value<std::string>(&output_file)->default_value(""), "Output CSV file for logging results (leave empty to disable logging)");
    boost::program_options::variables_map command_line_arguments;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, runtime_options), command_line_arguments);
    boost::program_options::notify(command_line_arguments);

    HEADER("CLI PARAMETERS: two-sided, host-initiated transfers between GPU and vFPGA");
    std::cout << "MODE: " << mode << std::endl;
    std::cout << "GPU ID: " << gpu_id << std::endl;
    std::cout << "Number of test runs: " << n_runs << std::endl;
    std::cout << "Number of transfers: " << n_transfers << std::endl;
    std::cout << "Starting transfer size: " << min_size << std::endl;
    std::cout << "Ending transfer size: " << max_size << std::endl << std::endl;

    // GPU memory will be allocated on the GPU set using hipSetDevice(...)
    if (hipSetDevice(gpu_id)) { throw std::runtime_error("Couldn't select GPU!"); }

    // Initialize ROCm SMI library
    PerfMonitor gpu_monitor(gpu_id);

    // Create a HIP stream; one for each parallel transfer
    std::vector<hipStream_t> hip_streams_d2h, hip_streams_h2d;
    for (unsigned int i = 0; i < n_transfers; i++) {
        hipStream_t stream_d2h;
        if (hipStreamCreate(&stream_d2h)) { throw std::runtime_error("Couldn't create D2H HIP stream!"); }
        hip_streams_d2h.emplace_back(stream_d2h);
    
        hipStream_t stream_h2d;
        if (hipStreamCreate(&stream_h2d)) { throw std::runtime_error("Couldn't create H2D HIP stream!"); }
        hip_streams_h2d.emplace_back(stream_h2d);
    }

    // Obtain a Coyote thread and allocate memory
    coyote::cThread coyote_thread(DEFAULT_VFPGA_ID, getpid());
    int *gpu_src, *gpu_dst;
    cpu_mem_pair_t cpu_src, cpu_dst;
    
    // P2P mode
    if (mode) {
        // Allocate GPU memory
        gpu_src = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, gpu_id});
        gpu_dst = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, gpu_id});
        if (!gpu_src || !gpu_dst) {  throw std::runtime_error("Could not allocate GPU memory; exiting..."); }

        // In P2P mode, there is no need for CPU memory
        cpu_src = {nullptr, nullptr};
        cpu_dst = {nullptr, nullptr};

    // Non-P2P mode
    } else {
        HIP_CHECK(hipMalloc((void **) &gpu_src, max_size));
        HIP_CHECK(hipMalloc((void **) &gpu_dst, max_size));
        if (!gpu_src || !gpu_dst) {  throw std::runtime_error("Could not allocate GPU memory; exiting..."); }
        
        // Use Coyote first to obtain the memory
        int* tmp_src_h = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::HPF, max_size});
        int* tmp_dst_h = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::HPF, max_size});

        // Register it with HIP to pin it
        if (hipHostRegister(tmp_src_h, max_size, hipHostRegisterDefault)) {
            throw std::runtime_error("Failed to pin CPU memory, exiting...");
        }

        if (hipHostRegister(tmp_dst_h, max_size, hipHostRegisterDefault)) {
            throw std::runtime_error("Failed to pin CPU memory, exiting...");
        }

        // Finally, obtain a pointer to the same memory, but from the GPU's point of view
        int *tmp_src_d = nullptr;
        int *tmp_dst_d = nullptr;
        if (hipHostGetDevicePointer((void **) &tmp_src_d, tmp_src_h, 0)) {
            throw std::runtime_error("Failed to obtain device pointer for CPU memory, exiting...");
        }
        if (hipHostGetDevicePointer((void **) &tmp_dst_d, tmp_dst_h, 0)) {
            throw std::runtime_error("Failed to obtain device pointer for CPU memory, exiting...");
        }

        // Save to struct
        cpu_src = {tmp_src_d, tmp_src_h}; 
        cpu_dst = {tmp_dst_d, tmp_dst_h};

        if (!cpu_src.first || !cpu_src.second || !cpu_dst.first || !cpu_dst.second) { throw std::runtime_error("Could not allocate CPU memory; exiting..."); }
    }

    // Finally allocate some memory that is used for setting the input and copying back the output from the GPU
    // Some platforms may support direct indexing of elements, e.g., gpu_src[i] = x; however, I am not a 100%
    // sure how that works and what performance (pinning, paging etc.) impacts it may have. Therefore, use the good,
    // old, fashioned-way of hipMemCpy(gpu_buff, cpu_buff)
    int *inputs, *results;
    HIP_CHECK(hipHostMalloc((void **) &inputs, max_size)); 
    HIP_CHECK(hipHostMalloc((void **) &results, max_size)); 
    if (!inputs || !results) { throw std::runtime_error("Could not allocate inputs/results memory; exiting..."); }

    HEADER("GPU <-> vFPGA PERFORMANCE");
    unsigned int curr_size = min_size;
    while(curr_size <= max_size) {
        // Run benchmark & calculate throughput
        PerfMetrics perf_metrics = run_bench(coyote_thread, hip_streams_d2h, hip_streams_h2d, gpu_src, gpu_dst, cpu_src, cpu_dst, inputs, results, curr_size, n_transfers, n_runs, mode, gpu_perf_monitoring, gpu_monitor);
        std::vector<double> measured_times = perf_metrics.get_all("latency");
        double avg_throughput = 0;
        for (const double &t : measured_times) {
            avg_throughput += ((double) n_transfers * (double) curr_size) / (1024.0 * 1024.0 * 1024.0 * t * 1e-9);;
        }
        avg_throughput = avg_throughput / (double) measured_times.size(); 

        // Print results
        std::cout << "Size: " << std::setw(8) << curr_size << "; ";
        std::cout << "Average latency: " << std::setw(8) << perf_metrics.get_average("latency") / 1e3 << " us; ";
        std::cout << "Average throughput: " << std::setw(8) << avg_throughput << " GB/s; ";
        if (gpu_perf_monitoring) {
            std::cout << "Average GPU power for latency test: " << std::setw(8) << perf_metrics.get_average("gpu_power") << " W; ";
            std::cout << "Average GPU utilization for latency test: " << std::setw(8) << perf_metrics.get_average("gpu_util") << " %; " << std::endl;
        } else {
            std::cout << std::endl;
        }

        // Log results to file if output file is specified
        if (!output_file.empty()) {
            log_to_file(
                output_file, 
                mode, 
                n_runs, 
                curr_size, 
                n_transfers,
                perf_metrics.get_average("latency") / 1e3,  // Convert to microseconds
                avg_throughput,
                perf_metrics.get_average("gpu_power"),
                perf_metrics.get_average("gpu_util")
            );
        }

        // Update size and proceed to next iteration
        curr_size *= 2;
    }

    // For the non-P2P case, do explicit memory de-allocated
    // For P2P, Coyote's destructor takes care of everything
    if (!mode) {
        HIP_CHECK(hipFree(gpu_src));
        HIP_CHECK(hipFree(gpu_dst));
    }

    // Free inputs/results buffer
    HIP_CHECK(hipHostFree(inputs));
    HIP_CHECK(hipHostFree(results));

    // Clean up HIP streams
    for (unsigned int i = 0; i < n_transfers; i++) {
        HIP_CHECK(hipStreamDestroy(hip_streams_d2h[i]));
        HIP_CHECK(hipStreamDestroy(hip_streams_h2d[i]));
    }

    return EXIT_SUCCESS;
}

/*
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

// External library for easier parsing of CLI arguments by the executable
#include <boost/program_options.hpp>

// Coyote-specific includes
#include <coyote/cBench.hpp>
#include <coyote/cThread.hpp>

// Constants
#define DEFAULT_VFPGA_ID 0
#define N_THROUGHPUT_REPS 16

// Registers, corresponding to registers defined the vFPGA (perf_fpga_axi_ctrl_parser.sv)
enum class BenchmarkRegisters: uint32_t {
    CTRL_REG = 0,           // AP start, read or write
    VADDR_REG = 1,          // Buffer virtual address
    LEN_REG = 2,            // Buffer length (size in bytes)
    PID_REG = 3,            // Coyote thread ID
    N_REPS_REG = 4,         // Number of read/write repetitions
    N_BEATS_REG = 5         // Number of expected AXI beats
};

// 01 written to CTRL_REG starts a read operation and 10 written to CTRL registers starts a write
enum class BenchmarkOperation: uint8_t {
    START_RD = 0x1,
    START_WR = 0x2
};

/**
 * @brief Executes a one-sided transfers (READ or WRITE) and measures DMA performance
 *
 * This function runs a series of one-sided read or write operations via the vFPGA,
 * measuring the time taken for each operation. 
 *
 * @param coyote_thread Reference to the Coyote thread for vFPGA communication
 * @param size Size of each transfer in bytes
 * @param mem Pointer to the GPU memory buffer to read from or write to
 * @param transfers Number of parallel transfers to launch in each operation
 * @param n_runs Number of actual benchmark runs to execute (after warm-up)
 * @param oper Benchmark operation type (START_RD for read, START_WR for write)
 * @return Vector of measured execution times in nanoseconds for each benchmark run
 */
std::vector<double> run_bench(
    coyote::cThread &coyote_thread, unsigned int size, int *mem, 
    unsigned int transfers, unsigned int n_runs, BenchmarkOperation oper
) {
    // Single iteration of transfers reads or writes
    auto benchmark_run = [&]() {
        coyote_thread.clearCompleted();
        uint64_t n_beats = transfers * ((size + 64 - 1) / 64);
        
        // Set the required registers from SW; the vFPGA uses these registers to start the transfer
        coyote_thread.setCSR(reinterpret_cast<uint64_t>(mem), static_cast<uint32_t>(BenchmarkRegisters::VADDR_REG));
        coyote_thread.setCSR(size, static_cast<uint32_t>(BenchmarkRegisters::LEN_REG));
        coyote_thread.setCSR(coyote_thread.getCtid(), static_cast<uint32_t>(BenchmarkRegisters::PID_REG));
        coyote_thread.setCSR(transfers, static_cast<uint32_t>(BenchmarkRegisters::N_REPS_REG));
        coyote_thread.setCSR(n_beats, static_cast<uint32_t>(BenchmarkRegisters::N_BEATS_REG));

        auto start_time = std::chrono::high_resolution_clock::now();
        coyote_thread.setCSR(static_cast<uint64_t>(oper), static_cast<uint32_t>(BenchmarkRegisters::CTRL_REG));

        // Poll on completion
        // NOTE: The hardware asserts the completion flag on the last beat; hence, there is only one completion (and not one per transfer)
        if (oper == BenchmarkOperation::START_RD) {
            while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ) != 1) {}
        } else {
            while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) != 1) {}            
        }

        // Capture time taken
        auto end_time = std::chrono::high_resolution_clock::now();
        return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    };

    // Warm-up runs
    for (int j = 0; j < 50; j++) {
        benchmark_run();
    }

    // Run benchmark
    std::vector<double> times;
    for (int j = 0; j < n_runs; j++) {
        double t = benchmark_run();
        times.push_back(t);
    }
    
    return times;
}

/**
 * @brief Utility function which stores the results of a benchmark to a CSV file
 *
 * @param file_name Name of the CSV file to store the results in (will be created if it doesn't exist, and appended to if it does exist)
 * @param operation READs (0) or WRITEs (1)
 * @param n_runs Number of times the test was repeated
 * @param size Size of the transfer in bytes
 * @param transfers Number of parallel transfers to launch
 * @param avg_throughput Average throughput of the transfers in GB/s
 */
void log_to_file(
    std::string file_name, bool operation, unsigned int n_runs, unsigned int size, unsigned int transfers,  double avg_throughput
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
        outfile << "timestamp,operation,n_runs,size,transfers,avg_throughput\n";
    }
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&now_time, &tm_buf);
    
    // Write data row with proper CSV formatting
    outfile << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << ","
            << operation << ","
            << n_runs << ","
            << size << ","
            << transfers << ","
            << avg_throughput << "\n";    
    outfile.close();
}

int main(int argc, char *argv[]) {
    // CLI arguments
    bool operation;
    std::string output_file;
    unsigned int gpu_id, n_runs, min_size, max_size;

    boost::program_options::options_description runtime_options("Coyote Perf FPGA Options");
    runtime_options.add_options()
        ("gpu_id,g", boost::program_options::value<unsigned int>(&gpu_id)->default_value(0), "Target GPU")
        ("output_file,f", boost::program_options::value<std::string>(&output_file)->default_value(""), "Output CSV file for logging results (leave empty to disable logging)")
        ("operation,o", boost::program_options::value<bool>(&operation)->default_value(false), "Benchmark operation: READ(0) or WRITE(1)")
        ("runs,r", boost::program_options::value<unsigned int>(&n_runs)->default_value(50), "Number of times to repeat the test")
        ("min_size,x", boost::program_options::value<unsigned int>(&min_size)->default_value(256), "Starting (minimum) transfer size")
        ("max_size,X", boost::program_options::value<unsigned int>(&max_size)->default_value(64 * 1024 * 1024), "Ending (maximum) transfer size");
    boost::program_options::variables_map command_line_arguments;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, runtime_options), command_line_arguments);
    boost::program_options::notify(command_line_arguments);
    BenchmarkOperation oper = operation ? BenchmarkOperation::START_WR : BenchmarkOperation::START_RD;

    HEADER("CLI PARAMETERS:");
    std::cout << "GPU ID: " << gpu_id << std::endl;
    std::cout << "Benchmark operation: " << (operation ? "WRITE" : "READ") << std::endl;
    std::cout << "Number of test runs: " << n_runs << std::endl;
    std::cout << "Starting transfer size: " << min_size << std::endl;
    std::cout << "Ending transfer size: " << max_size << std::endl << std::endl;

    if (hipSetDevice(gpu_id)) { throw std::runtime_error("Couldn't select GPU!"); }

    // Create Coyote thread and allocate source & destination memory
    coyote::cThread coyote_thread(DEFAULT_VFPGA_ID, getpid());
    int* mem =  (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, gpu_id});
    if (!mem) { throw std::runtime_error("Could not allocate memory; exiting..."); }

    // Benchmark sweep
    HEADER("BENCHMARK:");
    unsigned int curr_size = min_size;
    while (curr_size <= max_size) {
        // Run throughput test
        std::vector<double> measured_times = run_bench(coyote_thread, curr_size, mem, N_THROUGHPUT_REPS, n_runs, oper);
        double avg_throughput = 0;
        for (const double &t : measured_times) {
            avg_throughput += ((double) N_THROUGHPUT_REPS * (double) curr_size) / (1024.0 * 1024.0 * 1024.0 * t * 1e-9);;
        }
        avg_throughput = avg_throughput / (double) measured_times.size(); 

        // Print results
        std::cout << "Size: " << std::setw(8) << curr_size << "; ";
        std::cout << "Average throughput: " << avg_throughput << " GB/s; " << std::endl;

        // Log results to file if output file is specified
        if (!output_file.empty()) {
            log_to_file(
                output_file, 
                operation,
                n_runs, 
                curr_size, 
                N_THROUGHPUT_REPS,
                avg_throughput
            );
        }

        // Update size and proceed to next iteration
        curr_size *= 2;
    }

    return EXIT_SUCCESS;
}


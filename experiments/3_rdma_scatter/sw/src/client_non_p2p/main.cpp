/**
 * This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
 *
 * MIT Licence
 * Copyright (c) 2021-2025, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <cstdlib>
#include <fstream>

// AMD GPU management & run-time libraries
#include <hip/hip_runtime.h>

// External library for easier parsing of CLI arguments by the executable
#include <boost/program_options.hpp>

// Coyote-specific includes
#include "coyote/cBench.hpp"
#include "coyote/cThread.hpp"
#include "constants.hpp"

constexpr bool const IS_CLIENT = true;
const int NUM_GPUS = 4;

// Registers, corresponding to the registers defined in the vFPGA
enum class ScatterRegisters: uint32_t {
    VADDR_1 = 0, 
    VADDR_2 = 1, 
    VADDR_3 = 2, 
    VADDR_4 = 3, 
    VADDR_VALID = 4
}; 

// Note, how the Coyote thread is passed by reference; to avoid creating a copy of 
// the thread object which can lead to undefined behaviour and bugs. 
double run_bench(
    coyote::cThread &coyote_thread, coyote::rdmaSg &sg, 
    uint8_t *mem, int* dest_buffers[], uint transfers, uint n_runs, bool operation
) {

    // printf("Running benchmark with transfer size %d bytes, repeated %d times\n", sg.len, transfers);
    // When writing, the server asserts the written payload is correct (which the client sets)
    // When reading, the client asserts the read payload is correct (which the server sets)
    for (int i = 0; i < sg.len / sizeof(int); i++) {
        mem[i] = operation ? i : 0;         
    }
    
    // Before every benchmark, clear previous completion flags and sync with server
    // Sync is in a way equivalent to MPI_Barrier()
    auto prep_fn = [&]() {
        coyote_thread.clearCompleted();
        coyote_thread.connSync(IS_CLIENT);
    };
    
    /* Benchmark function; as eplained in the README
     * For RDMA_WRITEs, the client writes multiple times to the server and then the server writes the same content back
     * For RDMA READs, the client reads from the server multiple times
     * In boths cases, that means there will be n_transfers completed writes to local memory (LOCAL_WRITE)
     */
    coyote::CoyoteOper coyote_operation = operation ? coyote::CoyoteOper::REMOTE_RDMA_WRITE : coyote::CoyoteOper::REMOTE_RDMA_READ;
    auto bench_fn = [&]() {        
        for (int i = 0; i < transfers; i++) {
            coyote_thread.invoke(coyote_operation, sg);
        }

        while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) != transfers) {}

        // printf("Received all data from server, starting scatter to GPUs...\n");

        // After having received all the data, scatter it to the four GPU buffers 
        for(int i = 0; i < sg.len / 8192; i++) {
            // printf("Copying chunk %d/%d to GPU %d\n", i+1, sg.len/4096, i%4);
            // printf("Copying chunk %d/%d to GPU %d\n", i+1, sg.len/8192, i%NUM_GPUS);
            if (hipSetDevice(i%NUM_GPUS)) { throw std::runtime_error("Couldn't select GPU!"); }
            if (hipMemcpyAsync(dest_buffers[i % NUM_GPUS], &mem[i * 8192], 8192, hipMemcpyHostToDevice) != hipSuccess) { throw std::runtime_error("Couldn't copy memory to GPU!"); }
        }

        //printf("Issued all async copies to GPUs\n");

        // Sync mem copy for all the GPUs
        for(int i = 0; i < NUM_GPUS; i++) {
            if (hipSetDevice(i)) { throw std::runtime_error("Couldn't select GPU!"); }
            if (hipDeviceSynchronize() != hipSuccess) { throw std::runtime_error("Couldn't synchronize stream!"); }
        }
    };

    // Execute benchmark
    coyote::cBench bench(n_runs, 10);
    bench.execute(bench_fn, prep_fn);

    
    // For writes, divide by 2, since that is sent two ways (from client to server and then from server to client)
    // Reads are one way, so no need to scale
    return bench.getAvg() / (1. + (double) operation);
}

int main(int argc, char *argv[])  {
    // CLI arguments
    bool operation;
    std::string server_ip;
    unsigned int min_size, max_size, n_runs;
    bool throughput;

    boost::program_options::options_description runtime_options("Coyote Perf RDMA Options");
    runtime_options.add_options()
        ("ip_address,i", boost::program_options::value<std::string>(&server_ip), "Server's IP address")
        ("operation,o", boost::program_options::value<bool>(&operation)->default_value(false), "Benchmark operation: READ(0) or WRITE(1)")
        ("runs,r", boost::program_options::value<unsigned int>(&n_runs)->default_value(N_RUNS_DEFAULT), "Number of times to repeat the test")
        ("min_size,x", boost::program_options::value<unsigned int>(&min_size)->default_value(MIN_TRANSFER_SIZE_DEFAULT), "Starting (minimum) transfer size")
        ("max_size,X", boost::program_options::value<unsigned int>(&max_size)->default_value(MAX_TRANSFER_SIZE_DEFAULT), "Ending (maximum) transfer size")
        ("throughput,t", boost::program_options::bool_switch(&throughput)->default_value(false), "Whether to benchmark throughput (true) or latency (false); by default, latency is benchmarked");
    boost::program_options::variables_map command_line_arguments;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, runtime_options), command_line_arguments);
    boost::program_options::notify(command_line_arguments);

    HEADER("CLI PARAMETERS:");
    std::cout << "Server's TCP address: " << server_ip << std::endl;
    std::cout << "Benchmark operation: " << (operation ? "WRITE" : "READ") << std::endl;
    std::cout << "Number of test runs: " << n_runs << std::endl;
    std::cout << "Starting transfer size: " << min_size << std::endl;
    std::cout << "Ending transfer size: " << max_size << std::endl << std::endl;
    std::cout << "Benchmarking " << (throughput ? "throughput" : "latency") << std::endl << std::endl;

    /* Coyote completely abstracts the complexity behind exchanging QPs and setting up an RDMA connection
     * Instead, given a cThread, the target RDMA buffer size and the remote server's TCP address,
     * One can use the function initRDMA, which will allocate the buffer and 
     * Exchange the necessary information with the server; the server calls the equivalent function but without the IP address
     */
    coyote::cThread coyote_thread(DEFAULT_VFPGA_ID, getpid(), 0);
    uint8_t *mem = (uint8_t *) coyote_thread.initRDMA(max_size, coyote::DEF_PORT, server_ip.c_str());
    if (!mem) { throw std::runtime_error("Could not allocate memory; exiting..."); }

    // GPU memory will be allocated on the GPU set using hipSetDevice(...)
    // if (hipSetDevice(DEFAULT_GPU_ID)) { throw std::runtime_error("Couldn't select GPU!"); }
    
    // Allocate four buffers for the scatter operation 
    if (hipSetDevice(0)) { throw std::runtime_error("Couldn't select GPU!"); } 
    int* vaddr_1 = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, 0}); 
    int* vaddr_2 = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, 1}); 
    int* vaddr_3 = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, 2});
    int* vaddr_4 = (int *) coyote_thread.getMem({coyote::CoyoteAllocType::GPU, max_size, false, 3});

    // Print all the new buffer addresses
    std::cout << "Scatter buffer addresses:" << std::endl;
    std::cout << "Buffer 1: " << std::hex << reinterpret_cast<uint64_t>(vaddr_1) << std::endl;
    std::cout << "Buffer 2: " << std::hex << reinterpret_cast<uint64_t>(vaddr_2) << std::endl;
    std::cout << "Buffer 3: " << std::hex << reinterpret_cast<uint64_t>(vaddr_3) << std::endl;
    std::cout << "Buffer 4: " << std::hex << reinterpret_cast<uint64_t>(vaddr_4) << std::endl << std::dec;

    // Throw an exception if any of the buffers could not be allocated 
    if(!vaddr_1 || !vaddr_2 || !vaddr_3 || !vaddr_4) {
        throw std::runtime_error("Could not allocate memory for scatter buffers; exiting...");
    }

    // Write the buffer addresses to the vFPGA registers
    /* coyote_thread.setCSR(reinterpret_cast<uint64_t>(vaddr_1), static_cast<uint32_t>(ScatterRegisters::VADDR_1));
    coyote_thread.setCSR(reinterpret_cast<uint64_t>(vaddr_2), static_cast<uint32_t>(ScatterRegisters::VADDR_2));
    coyote_thread.setCSR(reinterpret_cast<uint64_t>(vaddr_3), static_cast<uint32_t>(ScatterRegisters::VADDR_3));
    coyote_thread.setCSR(reinterpret_cast<uint64_t>(vaddr_4), static_cast<uint32_t>(ScatterRegisters::VADDR_4));
    coyote_thread.setCSR(static_cast<uint64_t>(true), static_cast<uint32_t>(ScatterRegisters::VADDR_VALID)); */ 

    // Array of destination buffers for scatter operation
    int* dest_buffers[NUM_GPUS] = { vaddr_1, vaddr_2, vaddr_3, vaddr_4 };

    // sleep(20);

    // Benchmark sweep of latency and throughput
    HEADER("RDMA BENCHMARK: CLIENT");
    unsigned int curr_size = min_size;

    // Open a file to log the results
    std::ofstream baseline_results_file;
    if(throughput) {
        baseline_results_file.open("../../logs/baseline_thr.csv");
        if(!baseline_results_file.is_open()) {
            throw std::runtime_error("Couldn't open results file for writing!");
        } else {
            baseline_results_file << "avg_baseline \n";
        }
    } else {
        baseline_results_file.open("../../logs/baseline_lat.csv");
        if(!baseline_results_file.is_open()) {
            throw std::runtime_error("Couldn't open results file for writing!");
        } else {
            baseline_results_file << "avg_baseline \n";
        }
    }

    while(curr_size <= max_size) {
        std::cout << "Size: " << std::setw(8) << curr_size << "; ";
        
        coyote::rdmaSg sg = { .len = curr_size };
    
        if(throughput) {
            double throughput_time = run_bench(coyote_thread, sg, mem, dest_buffers, N_THROUGHPUT_REPS, n_runs, operation);
            double throughput = ((double) N_THROUGHPUT_REPS * (double) curr_size) / (1024.0 * 1024.0 * throughput_time * 1e-9);
            std::cout << "Average throughput: " << std::setw(8) << throughput << " MB/s; " << std::endl;
            baseline_results_file << throughput << std::endl;
        } else {
        
            double latency_time = run_bench(coyote_thread, sg, mem, dest_buffers, N_LATENCY_REPS, n_runs, operation);
            std::cout << "Average latency: " << std::setw(8) << latency_time / 1e3 << " us" << std::endl;
            baseline_results_file << latency_time << std::endl;
        } 

        curr_size *= 2;
    }

    // Final sync and exit
    coyote_thread.connSync(IS_CLIENT);
    return EXIT_SUCCESS;
}
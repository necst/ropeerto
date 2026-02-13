# 4. GPU - FPGA P2P micro-benchmarks

This directory contains the software and hardware source code for the results of Section 4 of the EuroSys'26 paper: *RoPeerTo: A Datacenter-Scale Architecture for Peer-To-Peer DMA between GPUs and FPGAs*.

The following is a brief guide on compiling and running this specific experiment presented in the paper. Most experiments consists of two folders: `hw` (hardware) and `sw` (software), both of which are built using `make`. For experiments described in Section 4, there are additional sub-folders: `two_sided` (results from Section 4.1, 4.2, 4.3) or `one_sided` (reults from Section 4.4) inside the folders `hw`/ `sw`. `two_sided` refers to host-initiated bidirectional (read & write) transfers, and hardware-initiated, `one_sided` (read or write) transfers. 

## Environment requirements & set-up
The following requirements must be met to run the experiments:

- Hardware:
    - AMD Alveo card. Experiments in the paper were conducted on an Alveo u55c. Other Coyote-supported platforms (e.g., u280, u250)  work as well.
    - AMD Instinct GPU. Experiments in the paper were conducted on a MI100. Other accelerator cards work (e.g., MI210) though results can vary.
    - For performance reasons, the FPGA and the GPU should be placed on the same NUMA node on multi-node systems. While RoPeerTo also works on multi-NUMA node systems, there is some performance degradation.

- Software/OS:
    - Linux >= 6 with DMABuff support and hugepages enabled
    - Vivado suite, with Vitis HLS, >= 2022.1 for synthesizing FPGA bitstreams
    - CMake >= 3.5, supporting C++17 standard
    - ROCm libraries. Experiments in the papers were conducted with ROCm 6.3.3. Important to note; in systems with more than one GPU, targetting other GPUs (i.e. gpu_id != 0), does not work in ROCm 6.3.3. If only ever targeting one (the first GPU), the following text can be ignored. This is a bug which was fixed during the development of the paper. Therefore, for ROCm 6.3.3, in addition it's necessary to apply the patch and re-install only the ROCR Runtime (but not the rest of the ROCm libraries). To apply the patch and re-install it, follow the steps below:
        1. Clone the ROCR Runtime, checking out the correct branch (matching the release of ROCm on the system): ```git clone -b release/rocm-rel-6.3.x https://github.com/ROCm/ROCR-Runtime.git```
        2. Apply the changes from the pull request: https://github.com/ROCm/ROCR-Runtime/pull/315 to the file `amd_gpu_agent.cpp`
        3. Install the run-time:
        ```bash
        mkdir build && cd build
        cmake -DCMAKE_INSTALL_PREFIX=</path/to/install/folder> -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
        make
        make install        
        ```
        4. If the installation completed correctly, `</path/to/install/folder>/lib` should contain a shared library file called `libhsa-runtime64.so`. Add it to your `LD_LIBRARY_PATH` as follows:
        ```
        export LD_LIBRARY_PATH=</path/to/install/folder>/lib/:$LD_LIBRARY_PATH
        ```
Experiments in the paper were conducted on the [AMD-ETH Heteregenous Accelerated Compute Cluster (HACC)](https://github.com/fpgasystems/hacc/tree/main), a publicly available cluster for research in systems, computer architecture, and accelerated applications. Its hardware equipment includes various compute nodes (Alveo U55C, V80, U250, U280, Instinct GPU etc.) which are connected via a high-speed (100G) network. With an account, the hardware can be synthesised on any of the build servers (hacc-build-01 or hacc-build-02) and the experiments can be executed on any of the HACC Boxes (nodes with a GPU and an FPGA; experiments in the paper were conducted on hacc-box-03, though the cluster set-up and its accompanying hardware may change at any point in the future without prior notice.)

**N.B.:** On hacc-build-01/02, Vivado 2024.1 can be loaded using `module load vivado/2024.1`.

**N.B.:** On the HACC Boxes, ROCm 6.3.3. can be loaded using `module load rocm/6.3.3`

## Synthesis & compilation
#### Hardware synthesis
Hardware builds can take hours, depending on the example complexity and synthesis flags. Therefore, if synthesizing on a remote node, it's recommended to ensure the process doesn't get terminated when the connection is lost, by using Linux utilities such as `screen` or `tmux`. To build the hardware, the following commands should be used:
```bash 
cd hw/two_sided                 # Change to hw/one_sided for experiments from Section 4.4.
mkdir build && cd build                
cmake ../ -DFDEV_NAME=u55c      # In the paper, we use the Alveo u55c; other Coyote-supported devices (u250, u280) work as well
make project && make bitgen
```

Once complete, a bitstream can be found in: `hw/two_sided/build/bitstreams/cyt_top.bit`. On the HACC, it is recommended to synthesize the hardware on one of the build nodes (hacc-build-01, hacc-build-02).

**N.B.:** Since hardware synthesis can take hours, we provide a pre-generated bitstream for the AMD Alveo u55c platform, found in `hw/two_sided/bitstream/cyt_top.bit` (or one-sided).

#### Driver compilation
The Coyote driver, which is required to interact with the FPGA, can be compiled with the following command:
```bash
cd Coyote/driver/
make
```

Once complete, a driver module can be foind in `Coyote/driver/build/coyote_driver.ko`

**N.B.:** The driver must always be compiled on the node used for running experiments (e.g., hacc-box-03), as the build and experiment nodes may have different versions of Linux.

#### Software compilation
The software compilation is largely similar to the hardware, but, typically much faster (typically completed within a minute).
```bash
cd sw/two_sided     # Change to sw/one_sided for experiments from Section 4.4.
mkdir build && cd build      
export CXX=hipcc          
cmake ../ -DEN_GPU=1
make
```

Once complete, a binary can be found in: `sw/two_sided/build/test`.

**N.B.:** The software must always be compiled on the node used for running experiments (e.g., hacc-box-03), as the build and experiment nodes may have different CPUs, leading to "illegal" instructions if running on a node which it wasn't compiled on.

**N.B.:** If using module, ensure ROCm is loaded before compiling the software. Additionally, with ROCm 6.3 or prior, ensure that the multi-GPU patch is applied if needed, as described above.

## Running the examples

#### Deploying the examples
To run the examples, it's necessary to load the bitstream and insert the driver. We sowec

**1. HACC:** On the HACC, the deployment is simplified through the `hdev` (HACC Development) tool, which allows us to easily program the FPGA and insert a driver. For this purpose, the script `util/program_hacc_local.sh` has been created:
```bash
bash util/program_hacc_local.sh <path-to-bitstream> <path-to-driver-ko>
```

**2. Independent set-up:** 
The steps to follow when deploying Coyote on an independent set-up are:
1. Program the FPGA using the synthesized bitstream using Vivado Hardware Manager via the GUI or a custom script

2. Rescan the PCIe devices and run PCI hot-plug.

3. Insert the driver using `sudo insmod <path-to-driver-ko> ip_addr=$qsfp_ip mac_addr=$qsfp_mac` (the parameters IP and MAC must only be specified when using networking on the FPGA, i.e. for experiment 3, RDMA Scatter). 

A successful completion of the FPGA programming and driver insertion can be checked via a call to
```bash
sudo dmesg
```

If the driver insertion and bitstream programming went correctly through, the last printed message should be `probe returning 0`. If you see this, your system is all ready to run the accompanying software.

#### Running the two-sided software (Sections 4.1 - 4.3)
Once the bitstream is loaded and the driver is inserted, the results presented in Sections 4.1 - 4.3 can be obtained by running the compiled two_sided software:
```bash
cd sw/two_sided/build     # Change to sw/one_sided for experiments from Section 4.4.
./test <arguments>
```

The available arguments are:
- `[--gpu_id | -g] <uint>` Target GPU (default: 0)
- `[--mode | -m] <bool>` Benchmark mode: 1 (P2P) or 0 (non-P2P, baseline) (default: 1)
- `[--gpu_perf_monitoring | -p] <bool>` Sample GPU power and utilization during the benchmark (default: 0)
- `[--runs | -r] <uint>` Number of times to repeat the test (default: 50)
- `[--transfers | -t] <uint>` Number of parallel transfers to launch (default: 1)
- `[--min_size | -x] <uint>` Starting (minimum) transfer size in bytes (default: 256)
- `[--max_size | -X] <uint>` Ending (maximum) transfer size in bytes (default: 2 MiB)
- `[--output_file | -f] <string>` Output CSV file for logging results (leave empty to disable logging) (default: "")


For example, the latency results from Section 4.1 can be obtained by running the following:
```bash
./test -f "latency.csv" -g 4 -m 1   # mode = 1 --> P2P
./test -f "latency.csv" -g 4 -m 0   # mode = 1 --> Baseline
```

And the throughput results from Section 4.2 can be obtained through the following commands (varying mode between P2P and non-P2P, as well as the number of parallel transfers):

```bash
# P2P benchmarks, scaling number of transfers between 2 and 16
./test -f "throughput.csv" -X 67108864 -g 4 -m 1 -t 2 
./test -f "throughput.csv" -X 67108864 -g 4 -m 1 -t 4 
./test -f "throughput.csv" -X 67108864 -g 4 -m 1 -t 8 
./test -f "throughput.csv" -X 67108864 -g 4 -m 1 -t 16

# Baseline benchmarks
./test -f "throughput.csv" -X 67108864 -g 4 -m 0 -t 2 
./test -f "throughput.csv" -X 67108864 -g 4 -m 0 -t 4 
./test -f "throughput.csv" -X 67108864 -g 4 -m 0 -t 8 
./test -f "throughput.csv" -X 67108864 -g 4 -m 0 -t 16
```

Finally, the power and GPU utilization for a 64 MiB transfer can be obtained as follows:
```bash
./test -f "power_util.csv" -x 67108864 -X 67108864 -g 4 -m 1 -p 1   # mode = 1 --> P2P; p = 1 --> collect GPU metrics
./test -f "power_util.csv" -x 67108864 -X 67108864 -g 4 -m 1 -p 1   # mode = 0 --> baseline
```

The files `results/latency.csv`, `results/throughput.csv` and `results/power_util.csv`, contain the respective results in a CSV format.

**N.B.:** At the time of paper writing, the experiments were conducted on hacc-box-03 which had 4 AMD Instinct MI210 GPUs (ID: 0-3) and a AMD Instinct MI100 GPU (ID: 4); the latter of which was used for the experiments. As the cluster set-up can change over time, ensure the parameter gpu_id is set correctly.

**N.B.:** On multi-NUMA sytems, it may be necessary to bind the program to the NUMA node of the executing GPU; for e.g., on hacc-box-03 where the MI100 resides on NUMA node #1, one should use 
```bash
numactl --cpunodebind=1 --membind=1 ./test <args>
```

#### Running the one-sided software (Sections 4.4)
Similary, the results from Section 4.4. can be reproduced with the one-sided software.

**N.B.:** Before running the one-sided experiments, ensure the FPGA is reprogrammed with the correct bitstram and the driver is re-insterted. The bitstream for Section 4.4 can be found in `hw/one_sided/build/bitstreams/cyt_top.bit`. The driver stays the same.

The arguments available for the one-sided software are:
- `[--gpu_id | -g] <uint>` Target GPU (default: 0)
- `[--output_file | -f] <string>` Output CSV file for logging results (leave empty to disable logging) (default: "")
- `[--operation | -o] <bool>` Benchmark operation: READ (0) or WRITE (1) (default: 0)
- `[--runs | -r] <uint>` Number of times to repeat the test (default: 50)
- `[--min_size | -x] <uint>` Starting (minimum) transfer size in bytes (default: 256)
- `[--max_size | -X] <uint>` Ending (maximum) transfer size in bytes (default: 64 MiB)

For example, to obtain the results from Section 4.4, run the following commands:
```bash
cd sw/one_sided/build
./test -f "read_write.csv" -g 4 -o 0    # Read operation
./test -f "read_write.csv" -g 4 -o 1    # Write operation
```
The file `results/read_write.csv` contains the results in CSV format.

**N.B.:** Same things to keep in mind as above, on GPU IDs and NUMA balancing.

#### Plotting the results
Utility scripts to recreate the plots from the paper can be found in the folder `scripts/`. For each script, it is necessary to pass the path to the correct results file (as obtained with the above-mentioned steps). The scripts parse the individual files and generate plots like the ones in the paper.

**N.B.:** The scripts rely on matplotlib, pandas and numpy which can be install via pip in a virtual environment.
```bash
python3 -m venv env
source env/bin/activate
pip install numpy matplotlib pandas
```

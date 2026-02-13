# Experiment 3: RDMA Scatter with RoPeerTo

This directory containts the software and hardware source code for the results of Section 4 of the EuroSys'26 paper: *RoPeerTo: A Datacenter-Scale Architecture for Peer-To-Peer DMA between GPUs and FPGAs*.

The general considerations about running the experiments as described in *1_microbenchmarks* also hold truth for this experiment, although the utilization of RDMA networking adds some further complexity: Most importantly, two FPGAs are required as *server* and *client* in the communication, and they need to be connected via 100G Ethernet, either directly or via RoCE-compatible switches. Furthermore, the two servers hosting the FPGAs need to have an additional network connection (i.e. via dedicated NICs) that are required for the QP-exchange and RoCE-communication establishment between the two FPGAs. The machine destinated as *client* needs to feature four distinct GPUs, preferrably on the same NUMA-node as the FPGA. The experiments for the paper have been conducted with 4 AMD MI210 GPUs. 

With regards to the experiments demonstrated in the paper, it is important to note that the *server* will always act as a "clean" RDMA-endpoint with access only to host memory. The *client* will then use RDMA READ operations to fetch data from this remote host and distribute it locally to the four connected GPUs, either through an offloaded scatter-mechanism implemented in the FPGA with direct GPU-access through RoPeerTo (P2P-experiment), or with local software-copies from host memory to the attached GPUs (baseline experiment). 

## Synthesis & compilation 
#### Hardware synthesis 
The principles of building bitstreams also apply to this experiment. Long building times can be avoided as we provide ready-made bitstreams for all use cases. However, the following rules apply for independent generation of the two required bitstreams:

1. *rdma_clean*: The "clean" bitstream (without offloaded scatter-logic) is to be used as server-node for both the P2P- and the baseline experiment and also as client-node for the baseline experiment. The following command triggers a build:

```bash
cd hw
mkdir build_rdma_clean && cd build_rdma_clean
cmake ../ -DFDEV_NAME=u55c -DINSTANCE=rdma_clean
make project && make bitgen
```

Once completed, the bitstream can be found in `hw/build_rdma_clean/bitstreams/cyt_top.bit`. Alternatively, the pre-generated bitstream is to be found in `hw/pregenerated_bitstreams/rdma_clean/cyt_top.bit`.

2. *rdma_scatter*: The scatter-bitstream (with offloaded scatter-logic) is used for the client in the P2P-experiment. The following command triggers a build:

```bash
cd hw
mkdir build_rdma_scatter && cd build_rdma_scatter
cmake ../ -DFDEV_NAME=u55c -DINSTANCE=rdma_scatter
make project && make bitgen
```

Once completed, the bitstream can be found in `hw/build_rdma_scatter/bitstreams/cyt_top.bit`. Alternatively, the pre-generated bitstream is to be found in `hw/pregenerated_bitstreams/rdma_scatter/cyt_top.bit`.

#### Software compilation
The same principles of software builds apply for the RDMA-experiment as for all other experiments in this artifact. For the two experiments, we need three software variants: The P2P-experiment features the *client_p2p* and *server*, the baseline experiment on the other hand *client_non_p2p* and *server*. These executables can be generated as following:

1. *client_p2p*: For building the P2P-based scatter-software, we execute the following commands:

```bash
cd sw
mkdir build_client_p2p && cd build_client_p2p
export CXX=hipcc
cmake ../ -DINSTANCE=client_p2p -DEN_GPU=1
make
```

The binary can be found in `sw/build_client_p2p/test`.

2. *client_non_p2p*: The following steps apply:

```bash
cd sw
mkdir build_client_non_p2p && cd build_client_non_p2p
export CXX=hipcc
cmake ../ -DINSTANCE=client_non_p2p -DEN_GPU=1
make
```

The binary can be found in `sw/build_client_non_p2p/test`.

3. *server*: The following steps apply:

```bash
cd sw
mkdir build_server && cd build_server
export CXX=hipcc
cmake ../ -DINSTANCE=server -DEN_GPU=1
make
```

The binary can be found in `sw/build_server/test`.


## Running the experiment 

#### Deploying the example 
The same steps as described in *1_microbenchmarks* apply in this case (flashing of bitstream, PCIe-rescan and driver insertion, preferrably through `program_hacc_local.sh`). It is important to note that this needs to be done independently for both the client and the server. The server will always deploy the *rdma_clean*-bitstream, while we switch between *rdma_scatter* and *rdma_clean* for the client in the different experiments (P2P vs. baseline).

#### Running the P2P-experiment: 
Since we deploy two nodes - *server* and *client* - we follow a two-step protocol. We start on the server-node (i.e. the machine where we previously loaded the *rdma_clean*-bitstream). There, we should first note down the out-of-band interface that we want to use for exchanging the QPs with the client. The IPv4-address of this interface can for example be found via `ifconfig`. Afterwards, we can start the server that will wait for the client to connect: 
```bash 
cd sw/build_server
./test -x 32768 -X 33554432 -r 30
```
Afterwards, we change to the client-machine (where we have ensured to have loaded the *rdma_scatter*-bitstream) and run the following commands: 
```bash 
cd sw/build_client_p2p
./test -x 32768 -X 33554432 -r 30 -i <"IPv4 address of the out-of-band-interface of the client">
```
The two machines will connect and run the experiment, afterwards the program on both machines should terminate gracefully. The given commands above will generate latency numbers. For getting throughput-measurements, the server needs to be started exactly the same way, while the client needs to be run with the additional argument `-t`. These two runs will log the outputs to `logs/p2p_lat.csv` and `logs/p2p_thr.csv` accordingly. 

Afterwards, we have the reflash the client with the *rdma_clean* bitstream, while the server remains unchanged. Following this scheme, we also start the server exactly the same way for the baseline experiment. For the client, we do the following:
```bash 
cd sw/build_client_non_p2p
./test -x 32768 -X 33554432 -r 30 -i <"IPv4 address of the out-of-band-interface of the client">
```
Again, we also run with the `-t` argument to obtain throughput values and generate finally both the `logs/baseline_lat.csv` and `logs/baseline_thr.csv`. 

#### Plotting the results
For generating the plots as shown in the paper, we switch to `logs` and execute the python-script `plot_results.py` without further arguments. It's important that all four mentioned log-files (baseline / p2p + lat / thr) are present in the directory, otherwise the execution of this script will fail. 
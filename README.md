# RoPeerTo: A Datacenter-Scale Architecture for Peer-To-Peer DMA between GPUs and FPGAs
This repositories contains the artifacts of the EuroSys '26 paper "RoPeerTo: A Datacenter-Scale Architecture for Peer-To-Peer DMA between GPUs and FPGAs". 
We introduce RoPeerTo, a novel architecture for peer-to-peer DMA between GPUs and FPGAs in datacenter environments. RoPeerTo enables direct data transfers between GPUs and FPGAs, bypassing the CPU and system memory, to achieve high performance and low latency. We design and implement RoPeerTo on a real hardware platform, and evaluate its performance using a set of microbenchmarks, and two real-world application: 3D medical image registration, and distributed computing with scatter-gather.


### Installation
We perform our experimental evaluation on the AMD-ETH Heterogenous Accelerated Compute Cluster (HACC). Further information are available [here](https://github.com/fpgasystems/hacc).
At the time of Artifact Submission, our setup is implemented on machine hacc-box-03.

For initial setup, please run:
 
```
export PATH=/usr/bin:$PATH
```

To clone the repository with all submodules
```
 git clone --recurse-submodules https://github.com/necst/ropeerto.git
```

To compile and run our experimental analysis, please refer to the README.md files corresponding to each directory.


### Repo Structure
The repository is structured as follows:

- `Coyote/`: contains the source code of Coyote, the FPGA architecture used in our work. Coyote is a novel FPGA architecture designed for datacenter environments, which provides a set of hardware services and vFPGAs for application acceleration. Coyote is open-source and can be used for research and development in FPGA-based acceleration.

- `experiments/`: contains the source code and artifacts of the experiments presented in the paper. Each experiment is organized in a separate folder, which contains the hardware and software source code, as well as instructions for building and running the experiment. The experiments include microbenchmarks for evaluating the performance of RoPeerTo, as well as two real-world applications: 3D medical image registration, and distributed computing with scatter-gather. Please refer to the README.md file for each experiment for further info.

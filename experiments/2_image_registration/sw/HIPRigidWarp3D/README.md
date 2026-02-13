### Note: 
This repo specifically regards HIP code for AMD GPUs. The CUDA version is in the orginal repository

# How to build and run
To build and run:
```shell
make run
```
With optional parameters:
```shell
make run [SIZE=<size>] [DEPTH=<depth>] [RUNS=<runs>] [TX=<tx>] [TY=<ty>] [ANG=<ang>]
```

To build only:
```shell
make all
```
Note: `Makefile` written for Windows. If you are using Linux, you may need to somehow fix the `Makefile`.
`

# Input/output volumes
Input and output volumes are stored in subfolders of `data/input` and `data/output` respectively. Automatically generated input volumes are stored in `data/input/generated` and are removed with `make clean`.

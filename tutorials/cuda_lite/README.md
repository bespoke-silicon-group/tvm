# Usage Guide for Running CUDA Lite Examples on HammerBlade
- Currently support running on AWS F1 and BSG XOR server.

## Installation
### Requirements
* BSG Bladerunner v3.2.1
* LLVM v7.0.1
* TVM (This repository)
* Python3

### Install BSG Bladerunner F1 instance
* Setup the BSG Bladerunner toolchain by cloning the [bsg_bladerunner](https://github.com/bespoke-silicon-group/bsg_bladerunner/tree/master) repo or lunch a bsg_bladerunner F1 instance.
* Compile the bsg_f1 cl_manycore library
* We currently support v3.2.1

### If you're using F1, install SCL devtools and python
* Install SCL [Developer Toolset 7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/)
* Install SCL [Python3.6](https://www.softwarecollections.org/en/scls/rhscl/rh-python36/)
* Add these two lines in your shell configuration file (e.g. ~/.bashrc) to enable it by default
```shell
source /opt/rh/devtoolset-7/enable
source /opt/rh/rh-python36/enable
```

### Install LLVM from source
* Download LLVM v7.1.0 source code from [here.](https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz)
* Follow the guide [here](https://llvm.org/docs/GettingStarted.html), and compile LLVM from source.

### Install TVM
* Clone this repository with
```shell
git clone -b hammerblade --recursive https://github.com/bespoke-silicon-group/tvm.git
```
* Follow the TVM installation [guide](https://docs.tvm.ai/install/from_source.html) to compile TVM from source.
* Please modify the config.cmake, set USE_CUDA_LITE to on and set USE_LLVM to llvm-config path.
```shell
set(USE_CUDA_LITE /path/to/bsg_bladerunner/root)
set(USE_LLVM /path/to/llvm-config)
```
* This code will link bsg_manycore_runtime library from the default linker path (/usr/lib64). If you wish to use to use custom built bsg_manycore_runtime library for the usage such as cosim, please compile the library in the bsg_f1 path, and add the following line to config.cmake.
```shell
set(BSG_F1_DIR /path/to/bsg_f1)
```
* Add the following lines to your shell configuration file(e.g. ~/.bashrc): 
```shell
export TVM_HOME=/path/to/tvm/
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:$TVM_HOME/hb/python:${PYTHONPATH}"
```
## Tutorials
If you're using AWS F1, please run setup_fpga.sh before running each tutorial script. 
setup_fpga.sh will initialize the FPGA with AGFI ID.
1. vec_add_cuda_lite.py
> Doing vector addition with cuda_lite, a good starting point to learn TVM dsl.
2. dot_product_cuda_lite.py
> Example code to do dot product, use more TVM dsl features.
3. gemm_cuda_lite.py
> General matrix multiplication is an important kernel for many other workloads(e.g. Convolutional Neural Network). This code use an easy to understand TVM scheduling that can run on manycore.
4. gemm_cuda_lite_ir_pass.py
> This code is based on the gemm_cuda_lite.py but apply an ir pass to detect and insert thread loop. So we can define workload with size larger than the underline hardware core.
5. relay_xxxx_cuda_lite.py
> There are the scripts testing the workloads defined in relay. The workloads are implemented in https://github.com/bespoke-silicon-group/tvm/tree/hammerblade/python/tvm/relay/testing.

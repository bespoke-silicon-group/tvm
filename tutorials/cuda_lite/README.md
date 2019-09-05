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
> These are the scripts testing the workloads defined in relay. The workloads are implemented in [here](https://github.com/bespoke-silicon-group/tvm/tree/hammerblade/python/tvm/relay/testing).

## Developer Guide
The TVM [Tutorials](https://docs.tvm.ai/tutorials/index.html) and the TVM [Design and Developer Guide](https://docs.tvm.ai/dev/index.html) are the good starting points to understand TVM usage and basic ideas of the code base.
Among these, https://docs.tvm.ai/dev/codebase_walkthrough.html and https://docs.tvm.ai/dev/runtime.html are the must read guides to understand TVM code structures quickly.

The developments for HammerBlade are mainly focus in three parts, runtime system, codegen and Topi operators support for HammerBlade. We will give detailed descriptions in the following sections.

### Runtime System for HammerBlade
The runtime system serves as the host code in the CUDA programming model.
With the help of the runtime system, TVM doesn't need to generate host code and can manage the device through the python interface.
The runtime system codebase for HammerBlade is in [here](https://github.com/bespoke-silicon-group/tvm/tree/hammerblade/src/runtime/hbmc).
The runtime system relies on the [bsg_f1](https://github.com/bespoke-silicon-group/bsg_f1) library and calls into the bsg_f1 library functions.

For the data allocation and data copy interface, please check [hbmc_device_api.cc](https://github.com/bespoke-silicon-group/tvm/blob/hammerblade/src/runtime/hbmc/hbmc_device_api.cc).
We follow the TVM defined device api and fullfill that with bsg_f1 library.

For kernel launch and passing arguments to the kernel, please check [hbmc_module.cc](https://github.com/bespoke-silicon-group/tvm/blob/hammerblade/src/runtime/hbmc/hbmc_module.cc).
The kernel launch in python will call back to the operator overloading functin of () at [here](https://github.com/bespoke-silicon-group/tvm/blob/1fee5a129a7f9d31fa34d1f6af1df9e7e1a40ebd/src/runtime/hbmc/hbmc_module.cc#L183).

### Codegen for HammerBlade (CUDA-Lite)
TVM codegen is a recursive process, it will call VisitStmt_() again and again with function overloading to traverse the IR syntax tree and print out the code for each node in the tree.
The CUDA-Lite code generation is similar to the CUDA code generation, and are implemented in the [codegen_cuda_lite.cc](https://github.com/bespoke-silicon-group/tvm/blob/hammerblade/src/codegen/codegen_cuda_lite.cc).
The recursive codegen process starts from this [function](https://github.com/bespoke-silicon-group/tvm/blob/1fee5a129a7f9d31fa34d1f6af1df9e7e1a40ebd/src/codegen/codegen_cuda_lite.cc#L79).

### TOPI Operators for HammerBlade
TVM Operator Inventory (TOPI) is the TVM implementation and scheduling for the often-use operators in deep learning.
You can find the introduction of TOPI at [here](https://docs.tvm.ai/tutorials/topi/intro_topi.html#sphx-glr-tutorials-topi-intro-topi-py).
For the HammerBlade, we can reuse the implemented operators, but we need to write new schedulings to map the computation correctly to the ManyCore hardware.
The redefined CUDA-Lite schedulings are implemented in the [topi/python/topi/cuda](https://github.com/bespoke-silicon-group/tvm/tree/1fee5a129a7f9d31fa34d1f6af1df9e7e1a40ebd/topi/python/topi/cuda) directory, since CUDA-Lite scheduling is generally similar to original CUDA scheduling.

To learn more about the TVM scheduling and optimization, please refer to the guides [here](https://docs.tvm.ai/tutorials/index.html#tensor-expression-and-schedules) and [here](https://docs.tvm.ai/tutorials/index.html#optimize-tensor-operators).
If you want to generate more optimized code for HammerBlade, the TOPI scheduling is the place you want to modify.

# Usage Guide for Running CUDA Lite Examples on HammerBlade
- Currently only support running on F1 instance

## Installation
### Requirements
* BSG Bladerunner F1 instance v1.0.2
* LLVM v7.1.0
* TVM (This repository)

### Install BSG Bladerunner F1 instance
* Follow the [guide](https://github.com/bespoke-silicon-group/bsg_bladerunner/tree/master) 
and launch a F1 instance with pre-built bsg_f1 and bsg_manycore.
* We current support v1.0.2.

### Install LLVM
* On the F1 instance, download LLVM v7.1.0 source code from [here.](https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz)
* Follow the guide [here](https://llvm.org/docs/GettingStarted.html), and compile LLVM from source.
* Notice: you may need to upgrade cmake version to compile LLVM.

### Install TVM
* Clone this repository and follow the TVM installation [guide](https://docs.tvm.ai/install/from_source.html) to compile TVM from source.
* Please modify the config.cmake, set USE_HBMC to on and set USE_LLVM to llvm-config path.
* Compile the bgs_f1 library and adjust the path of HBMC_HBMC_LIBRARY in CMakeLists.txt

## Run
* Run setup_fpga.sh to get the FPGA ready
* Enter root mode in order to execute the aws library. (you may need to add the python path for the root user)
* Execute the scripts here with python

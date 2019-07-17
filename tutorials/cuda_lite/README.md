# Usage Guide for Running CUDA Lite Examples on HammerBlade
- Currently only support running on F1 instance

## Installation
### Requirements
* BSG Bladerunner F1 instance v3.0.6
* LLVM v7.0.1
* TVM (This repository)
* Python3

### Install BSG Bladerunner F1 instance
* Follow the [guide](https://github.com/bespoke-silicon-group/bsg_bladerunner/tree/master) 
and launch a F1 instance with pre-built bsg_f1 and bsg_manycore.
* Compile the bsg_f1 cl_manycore library
* We current support v3.0.6.

### Install newer version of devtools and python on AWS
* Install SCL [Developer Toolset 7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/)
* Install SCL [Python3.6](https://www.softwarecollections.org/en/scls/rhscl/rh-python36/)
* Add these two lines in your shell configuration file (e.g. ~/.bashrc) to enable it by default
```shell
source /opt/rh/devtoolset-7/enable
source /opt/rh/rh-python36/enable
```

### Install LLVM
* On the F1 instance, download LLVM v7.1.0 source code from [here.](https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz)
* Follow the guide [here](https://llvm.org/docs/GettingStarted.html), and compile LLVM from source.

### Install TVM
* Clone this repository with
```shell
git clone --recursive https://github.com/bespoke-silicon-group/tvm.git
git fetch
git checkout hammerblade
```
* Follow the TVM installation [guide](https://docs.tvm.ai/install/from_source.html) to compile TVM from source.
* Please modify the config.cmake, set USE_HBMC to on and set USE_LLVM to llvm-config path.
* Compile the bgs_f1 library and adjust the path of HBMC_HBMC_LIBRARY in CMakeLists.txt.

## Run
* Run setup_fpga.sh to get the FPGA ready
* Enter root mode in order to execute the aws library. (you may need to add the python path for the root user)
* Execute the scripts here with python

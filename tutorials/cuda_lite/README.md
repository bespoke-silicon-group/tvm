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
* We current support v3.2.1

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
* Compile the bgs_f1 library.
* Add the following lines to your shell configuration file(e.g. ~/.bashrc): 
```shell
export TVM_HOME=/path/to/tvm/
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}"
```
## Run
* Run setup_fpga.sh to get the FPGA ready
* Enter root mode in order to execute the aws library. (you may need to add the python path for the root user)
* Execute the scripts in tutorials/cuda_lite with python

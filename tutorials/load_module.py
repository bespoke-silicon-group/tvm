"""
Get Started with TVM
====================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introduction tutorial to TVM.
TVM is a domain specific language for efficient kernel construction.

In this tutorial, we will demonstrate the basic workflow in TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"

'''
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))
'''

# s = tvm.create_schedule(C.op)

# bx, tx = s[C].split(C.op.axis[0], factor=64)

'''
if tgt == "cuda" or tgt.startswith('opencl'):
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
'''

# fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

######################################################################
# Run the Function
# ----------------
# The compiled function TVM function is designed to be a concise C API
# that can be invoked from any languages.
#
# We provide an minimum array API in python to aid quick testing and prototyping.
# The array API is based on `DLPack <https://github.com/dmlc/dlpack>`_ standard.
#
# - We first create a gpu context.
# - Then tvm.nd.array copies the data to gpu.
# - fadd runs the actual computation.
# - asnumpy() copies the gpu array back to cpu and we can use this to verify correctness
#
ctx = tvm.context(tgt, 0)

n = 1024
'''
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
'''
# fadd(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Inspect the Generated Code
# --------------------------
# You can inspect the generated code in TVM. The result of tvm.build
# is a tvm Module. fadd is the host module that contains the host wrapper,
# it also contains a device module for the CUDA (GPU) function.
#
# The following code fetches the device module and prints the content code.
#
'''
if tgt == "cuda" or tgt.startswith('opencl'):
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
else:
    print(fadd.get_source())
'''

######################################################################
# .. note:: Code Specialization
#
#   As you may noticed, during the declaration, A, B and C both
#   takes the same shape argument n. TVM will take advantage of this
#   to pass only single shape argument to the kernel, as you will find in
#   the printed device code. This is one form of specialization.
#
#   On the host side, TVM will automatically generate check code
#   that checks the constraints in the parameters. So if you pass
#   arrays with different shapes into the fadd, an error will be raised.
#
#   We can do more specializations. For example, we can write
#   :code:`n = tvm.convert(1024)` instead of :code:`n = tvm.var("n")`,
#   in the computation declaration. The generated function will
#   only take vectors with length 1024.
#

######################################################################
# Save Compiled Module
# --------------------
# Besides runtime compilation, we can save the compiled modules into
# file and load them back later. This is called ahead of time compilation.
#
# The following code first does the following step:
#
# - It saves the compiled host module into an object file.
# - Then it saves the device module into a ptx file.
# - cc.create_shared calls a env compiler(gcc) to create a shared library
#
from tvm.contrib import cc
from tvm.contrib import util

'''
temp = util.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt.startswith('opencl'):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())
'''

######################################################################
# .. note:: Module Storage Format
#
#   The CPU(host) module is directly saved as a shared library(so).
#   There can be multiple customized format on the device code.
#   In our example, device code is stored in ptx, as well as a meta
#   data json file. They can be loaded and linked separately via import.
#

######################################################################
# Load Compiled Module
# --------------------
# We can load the compiled module from the file system and run the code.
# The following code load the host and device module separately and
# re-link them together. We can verify that the newly loaded function works.
#
so_path = "/home/centos/bsg_f1/cl_manycore/software/deploy/read_write_test_no_main.so"
# fadd1 = tvm.module.load(temp.relpath("myadd.so"))
fadd1 = tvm.module.load(so_path)
'''
if tgt == "cuda":
    fadd1_dev = tvm.module.load(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt.startswith('opencl'):
    fadd1_dev = tvm.module.load(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)
'''

exit()
fadd1(a, b, c)
#tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Pack Everything into One Library
# --------------------------------
# In the above example, we store the device and host code separately.
# TVM also supports export everything as one shared library.
# Under the hood, we pack the device modules into binary blobs and link
# them together with the host code.
# Currently we support packing of Metal, OpenCL and CUDA modules.
#
fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.module.load(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())


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

tgt_host="llvm"
# Set target to HammerBlade Manycore
tgt="hbmc"

n = tvm.var("n")
A = tvm.placeholder((n,), dtype='int', name='A')
B = tvm.placeholder((n,), dtype='int', name='B')
# Describe the computation using TVM DSL
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

# Set target to HammerBlade Manycore
tgt="hbmc"
ctx = tvm.context(tgt, 0)

# Initialize data on the host
n = 16 # the length of the vector
a = tvm.nd.array(np.random.uniform(low=0, high=10, size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(low=0, high=10, size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)

# Load the host code binary
fadd1 = tvm.module.load("./myadd_int.so")
if tgt == "hbmc": # Load the device code binary
    fadd1_dev = tvm.module.load("./myadd.riscv")
    fadd1.import_module(fadd1_dev) 

# Launch the kernel on the device
fadd1(a, b, c)
# Check the final results
print("a: " + str(a.asnumpy()))
print("b: " + str(b.asnumpy()))
print("c = a + b = " + str(c.asnumpy()))
#tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Pack Everything into One Library
# --------------------------------
# In the above example, we store the device and host code seperatedly.
# TVM also supports export everything as one shared library.
# Under the hood, we pack the device modules into binary blobs and link
# them together with the host code.
# Currently we support packing of Metal, OpenCL and CUDA modules.
#
# fadd.export_library(temp.relpath("myadd_pack.so"))
# fadd2 = tvm.module.load(temp.relpath("myadd_pack.so"))
# fadd2(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())


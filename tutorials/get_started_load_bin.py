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
tgt="hbmc"

######################################################################
# Vector Add Example
# ------------------
# In this tutorial, we will use a vector addition example to demonstrate
# the workflow.
#

######################################################################
# Describe the Computation
# ------------------------
# As a first step, we need to describe our computation.
# TVM adopts tensor semantics, with each intermediate result
# represented as multi-dimensional array. The user need to describe
# the computation rule that generate the tensors.
#
# We first define a symbolic variable n to represent the shape.
# We then define two placeholder Tensors, A and B, with given shape (n,)
#
# We then describe the result tensor C, with a compute operation.
# The compute function takes the shape of the tensor, as well as a lambda function
# that describes the computation rule for each position of the tensor.
#
# No computation happens during this phase, as we are only declaring how
# the computation should be done.
#
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
# print(type(C))


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
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
# fadd(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())


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
# Load Compiled Module
# --------------------
# We can load the compiled module from the file system and run the code.
# The following code load the host and device module seperatedly and
# re-link them together. We can verify that the newly loaded function works.
#

# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="hbmc"

#fadd1 = tvm.module.load(temp.relpath("myadd.so"))
fadd1 = tvm.module.load("./myadd.so")
#exit()
if tgt == "hbmc":
    fadd1_dev = tvm.module.load("./myadd.hbmc")
    #fadd1.import_module(fadd1_dev) 

fadd1(a, b, c)
exit()
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

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


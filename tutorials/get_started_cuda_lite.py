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
tgt="cuda_lite"

#
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

#
s = tvm.create_schedule(C.op)

#
bx, tx = s[C].split(C.op.axis[0], factor=64)

######################################################################
# Finally we bind the iteration axis bx and tx to threads in the GPU
# compute grid. These are GPU specific constructs that allows us
# to generate code that runs on GPU.
#
if tgt == "cuda_lite" or tgt.startswith('opencl'):
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

#
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

#
#ctx = tvm.context(tgt, 0)
#
#n = 1024
#a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
#b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
#c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
#fadd(a, b, c)
#tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Inspect the Generated Code
# --------------------------
# You can inspect the generated code in TVM. The result of tvm.build
# is a tvm Module. fadd is the host module that contains the host wrapper,
# it also contains a device module for the CUDA (GPU) function.
#
# The following code fetches the device module and prints the content code.
#
if tgt == "cuda_lite" or tgt.startswith('opencl'):
    dev_module = fadd.imported_modules[0]
    print("-----CUDA Lite code-----")
    print(dev_module.get_source())
else:
    print(fadd.get_source())

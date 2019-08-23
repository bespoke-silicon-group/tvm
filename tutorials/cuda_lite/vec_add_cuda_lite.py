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
import sys

# Global declarations of environment.

tgt_host="llvm"
# Use cuda_litr or hbmc to generate code for hammerblade
tgt="cuda_lite"

dtype = 'float32'

n = tvm.var("n")
A = tvm.placeholder((n,), dtype=dtype, name='A')
B = tvm.placeholder((n,), dtype=dtype, name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=2)

if tgt == "cuda_lite" or tgt == "hbmc":
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

if tgt == "cuda_lite" or tgt.startswith('opencl'):
    dev_module = fadd.imported_modules[0]
    print("-----CUDA Lite code-----")
    print(dev_module.get_source())

ctx = tvm.context(tgt, 0)

np.set_printoptions(threshold=sys.maxsize)
n = 16
#a = tvm.nd.array(np.random.randint(10, size=n).astype(A.dtype), ctx)
a = tvm.nd.array(np.random.uniform(-1, 1, size=n).astype("float32"), ctx)
print(a)
#b = tvm.nd.array(np.random.randint(10, size=n).astype(B.dtype), ctx)
b = tvm.nd.array(np.random.uniform(-1, 1, size=n).astype("float32"), ctx)
print(b)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
print(c)
print(a.asnumpy() + b.asnumpy())
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy(), rtol=1e-3, atol=1e-2)


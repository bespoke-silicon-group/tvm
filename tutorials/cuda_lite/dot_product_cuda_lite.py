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
import time

# Global declarations of environment.

tgt_host="llvm"
# Use cuda_litr or hbmc to generate code for hammerblade
tgt="cuda_lite"

dtype = "float32"

n = tvm.var("n")
k = tvm.reduce_axis((0, n), "k")
A = tvm.placeholder((n, ), dtype=dtype, name='A')
B = tvm.placeholder((n, ), dtype=dtype, name='B')
C = tvm.compute((1, ), lambda i: tvm.sum(A[k] * B[k], axis=k), name="C")

s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=4)
#bx, tx = s[C].split(C.op.axis[0], nparts=1)

if tgt == "cuda_lite" or tgt == "hbmc":
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

fdot = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="mydot")

if tgt == "cuda_lite":
    dev_module = fdot.imported_modules[0]
    print("-----CUDA Lite code-----")
    print(dev_module.get_source())

ctx = tvm.context(tgt, 0)
n = 32768*2*2
#np.random.seed(int(time.time()))
a = tvm.nd.array(np.random.uniform(-1, 1, size=n).astype("float32"), ctx)
print(a)
b = tvm.nd.array(np.random.uniform(-1, 1, size=n).astype("float32"), ctx)
print(b)
c = tvm.nd.array(np.zeros(1, dtype=C.dtype), ctx)
fdot(a, b, c)
print(c)
print(sum(a.asnumpy()*b.asnumpy()))
if not tvm.testing.assert_allclose(c.asnumpy(), np.sum(a.asnumpy()*b.asnumpy()), rtol=1e-05):
    print("CUDA-Lite RESULTS MATCH CPU RESULTS (WITHIN TOLERANCE)")


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
#from ir_pass import inject_thread_loop
from hb import ir_pass

# Global declarations of environment.
dtype = 'int32'

M = 32

#n = tvm.var("n")
k = tvm.reduce_axis((0, M), 'k')
A = tvm.placeholder((M, M), name="A", dtype=dtype)
B = tvm.placeholder((M, M), name="B", dtype=dtype)

C = tvm.compute(
        (M, M),
        lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
        name='C')

scale = 4
num_thread = 4
block_factor = scale*num_thread

s = tvm.create_schedule(C.op)

block_x = tvm.thread_axis("blockIdx.x")
thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
block_y = tvm.thread_axis("blockIdx.y")
thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")

by, yi = s[C].split(C.op.axis[0], factor=block_factor)
bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
s[C].bind(by, block_y)
s[C].bind(bx, block_x)
s[C].reorder(by, bx, yi, xi)

ty, yi = s[C].split(yi, nparts=num_thread)
tx, xi = s[C].split(xi, nparts=num_thread)
s[C].bind(ty, thread_y)
s[C].bind(tx, thread_x)
s[C].reorder(ty, tx, yi, xi)

tgt_host="llvm"
# Use cuda_litr or hbmc to generate code for hammerblade
tgt="cuda_lite"

# Build the hammerblade module with ir pass for cuda lite
with tvm.build_config(add_lower_pass=[(1, ir_pass.inject_thread_loop)]):
    func = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="cuda_lite_gemm")

if tgt == "cuda_lite" or tgt.startswith('opencl'):
    dev_module = func.imported_modules[0]
    print("-----CUDA Lite code-----")
    print(dev_module.get_source())
#exit()

ctx = tvm.context(tgt, 0)

a_np = np.random.randint(10, size=(M ,M)).astype(dtype)
b_np = np.random.randint(10, size=(M ,M)).astype(dtype)
c_np = np.zeros((M, M), dtype=C.dtype)

a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(c_np, ctx)
func(a, b, c)
ans = np.dot(a.asnumpy(), b.asnumpy())
err = tvm.testing.assert_allclose(c.asnumpy(), ans, rtol=1e-5)
if not err:
    print("The matrix multiplication results are the same as numpy.")

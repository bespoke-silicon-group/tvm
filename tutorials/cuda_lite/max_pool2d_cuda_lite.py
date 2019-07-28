"""
Get Started with TVM
====================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introduction tutorial to TVM.
TVM is a domain specific language for efficient kernel construction.

In this tutorial, we will demonstrate the basic workflow in TVM.
"""
from __future__ import absolute_import, print_function

BSG_CRED   = '\033[91m'
BSG_CGREEN = '\033[92m'
BSG_CEND   = '\033[0m'

import tvm
import numpy as np





# Global declarations of environment.
dtype = 'int32'

M = 16
N = 4
sub_block = M // N
n = tvm.var("n")


# Simple max_pool2d on the host side to compare results
def host_max_pool2d(a):
    ans = [[0 for x in range(N)] for y in range(N)] 
    for i in range(0,N):
        for j in range (0,N):
            maxx = -1;
            for k in range (0,sub_block):
                for p in range (0,sub_block):
                    maxx = max(a[i * sub_block + k][j * sub_block + p], maxx)
            ans[i][j] = maxx
    return ans






k = tvm.reduce_axis((0, sub_block), 'k')
j = tvm.reduce_axis((0, sub_block), 'j')
A = tvm.placeholder((M, M), name="A", dtype=dtype)
B = tvm.compute(
        (N, N),
        lambda x, y: tvm.max(A[(x * sub_block + k), (y * sub_block + j)], axis=[k, j]),
        name='B')



scale = 4
num_thread = 1
block_factor = scale*num_thread

s = tvm.create_schedule(B.op)

block_x = tvm.thread_axis("blockIdx.x")
thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
block_y = tvm.thread_axis("blockIdx.y")
thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")

by, yi = s[B].split(B.op.axis[0], factor=block_factor)
bx, xi = s[B].split(B.op.axis[1], factor=block_factor)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)
s[B].reorder(by, bx, yi, xi)

ty, yi = s[B].split(yi, nparts=num_thread)
tx, xi = s[B].split(xi, nparts=num_thread)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)
s[B].reorder(ty, tx, yi, xi)

tgt_host="llvm"
# Use cuda_lite or hbmc to generate code for hammerblade
tgt="cuda_lite"

func = tvm.build(s, [A, B], tgt, target_host=tgt_host, name="cuda_lite_max_pool2d")

if tgt == "cuda_lite" or tgt.startswith('opencl'):
    dev_module = func.imported_modules[0]
    print("-----CUDA Lite code-----")
    print(dev_module.get_source())

ctx = tvm.context(tgt, 0)

a_np = np.random.randint(10, size=(M ,M)).astype(dtype)
b_np = np.zeros((N, N), dtype=B.dtype)

a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
func(a, b)
print (a)
print (b)

ans = host_max_pool2d(a_np)

print(ans)

err = tvm.testing.assert_allclose(b.asnumpy(), ans, rtol=1e-5)
if not err:
    print(BSG_CGREEN + "PASS: Max Pool 2D results are correct." + BSG_CEND )
else:
    print(BSG_CRED + "FAIL: Max Pool 2D results are incorrect." + BSG_CEND )


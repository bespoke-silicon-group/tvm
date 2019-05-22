from __future__ import absolute_import, print_function
import tvm
import numpy as np

######################################################################
# We first write a very simple vector add and build it with the default schedule. Then, we use
# our customized lowering pass to manipulate the IR directly instead of using schedule primitives.
#
n = tvm.const(128, "int32")
A = tvm.placeholder((n, ), name="A")
B = tvm.placeholder((n, ), name="B")

##### Creating vector addition IR for regular GPU computation #####
def gpu_vec_add_ir(A, B, C):
    max_threads = 32
    ib = tvm.ir_builder.create()

    bx = tvm.thread_axis("blockIdx.x")
    tx = tvm.thread_axis("threadIdx.x")

    ib.scope_attr(bx, "thread_extent", (n+max_threads-1)//max_threads)
    ib.scope_attr(tx, "thread_extent", max_threads)

    idx = bx.var * max_threads + tx.var

    Aptr = ib.buffer_ptr(A)
    Bptr = ib.buffer_ptr(B)
    Cptr = ib.buffer_ptr(C)

    with ib.if_scope(ib.likely(idx<n)):
        Cptr[idx] = Aptr[idx] + Bptr[idx]

    body = ib.get()
    return body

C = tvm.extern(A.shape, [A, B], lambda inps, outs: gpu_vec_add_ir(inps[0], inps[1], outs[0]), 
               name="vec_add", dtype="float32")

s = tvm.create_schedule(C.op)
bounds = tvm.schedule.InferBound(s)
stmt = tvm.schedule.ScheduleOps(s, bounds)
ir  = tvm.lower(s, [A, B, C], simple_mode=True)
print("Original IR")
print(ir)

threads = []
def find_thread4(op):
    if isinstance(op, tvm.stmt.AttrStmt):
        if op.attr_key == "thread_extent" and str(op.node.var) == "threadIdx.x":
            if op.value.value > 4:
                threads.append(op)

##### IR Transformation for manycore usage #####
def loopify4(op):
    """ Add a thread loop for the tasks found in `find_thread4`. """
    if op in threads:
        tx_num = op.value.value
        li = tvm.var('i')
        if isinstance(op.body, tvm.stmt.Store):
            body = tvm.ir_pass.Substitute(op, {op.body.index.b: op.body.index.b*(tx_num//4) + li})
            body = tvm.make.For(li, 0, tx_num // 4, tvm.stmt.For.Serial, 0, body)
        return body
    return None

def thread_loop(stmt):
    global threads

    tvm.ir_pass.PostOrderVisit(stmt, find_thread4)

    if not threads:
        return stmt

    stmt = tvm.ir_pass.IRTransform(stmt, None, loopify4, ['AttrStmt'])

    return stmt

print("Transformed IR")
print(thread_loop(ir))

with tvm.build_config(add_lower_pass=[(1, thread_loop)]) as cfg:
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    f = tvm.lower(s, [A, B, C], simple_mode=False)

fadd = tvm.build(f, target='cuda')
print("Generated Code")
print(fadd.imported_modules[0].get_source())


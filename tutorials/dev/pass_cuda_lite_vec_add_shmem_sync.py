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
max_thds = 32
num_thds = 4

##### Creating vector addition IR for regular GPU computation #####
def gpu_vec_add_ir_shmem(A, B, C):
    ib = tvm.ir_builder.create()

    bx = tvm.thread_axis("blockIdx.x")
    tx = tvm.thread_axis("threadIdx.x")

    ib.scope_attr(bx, "thread_extent", (n+max_thds-1)//max_thds)
    ib.scope_attr(tx, "thread_extent", max_thds)

    AA = ib.allocate("float32", max_thds, name="AA", scope="shared")
    BB = ib.allocate("float32", max_thds, name="BB", scope="shared")

    Aptr = ib.buffer_ptr(A)
    Bptr = ib.buffer_ptr(B)
    Cptr = ib.buffer_ptr(C)

    idx = bx.var * max_thds + tx.var
    with ib.if_scope(ib.likely(idx<n)):
        AA[tx.var] = Aptr[idx]
        BB[tx.var] = Bptr[idx]
        ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                              tvm.convert(['shared']),
                              tvm.expr.Call.Intrinsic, None, 0))
        Cptr[idx] = AA[tx.var] + BB[tx.var]

    body = ib.get()
    print(body)
    return body

print("Original IR")
C = tvm.extern(A.shape, [A, B], lambda inps, outs: gpu_vec_add_ir_shmem(inps[0], inps[1], outs[0]), 
               name="vec_add", dtype="float32")

s = tvm.create_schedule(C.op)
#bounds = tvm.schedule.InferBound(s)
#stmt = tvm.schedule.ScheduleOps(s, bounds)
ir  = tvm.lower(s, [A, B, C], simple_mode=True)
#print(ir)
#exit()

threads = []
def find_thread_loop_point(op):
    if isinstance(op, tvm.stmt.AttrStmt):
        if op.attr_key == "thread_extent" and str(op.node.var) == "threadIdx.x":
            if op.value.value > num_thds:
                threads.append(op)

##### IR Transformation for manycore usage #####
tx_num = None
li = tvm.var('i')

src_idx = tvm.var()
tar_idx = tvm.var()
def substitute_index(op):
    if isinstance(op.index, tvm.expr.Var):
        if str(op.index) == str(src_idx):
            body = tvm.ir_pass.Substitute(op, {op.index: tar_idx})
            return body
    if isinstance(op.index, tvm.expr.Add):
        if str(op.index.a) == str(src_idx):
            body = tvm.ir_pass.Substitute(op, {op.index.a: tar_idx})
            return body
        if str(op.index.b) == str(src_idx):
            body = tvm.ir_pass.Substitute(op, {op.index.b: tar_idx})
            return body
    return None

def inject_for_on_leaf_stmt(op):
    if isinstance(op, tvm.stmt.Stmt):
        # the Evaluate and Store are the only two types of leaf stmt nodes
        if isinstance(op, tvm.stmt.Store):
            body = tvm.make.For(li, 0, max_thds // num_thds, tvm.stmt.For.Serial, 0, op)
            body = tvm.ir_pass.IRTransform(body, None, substitute_index, ['Store'])
            return body

    return None

def merge_for_loops(op):
    if isinstance(op.first, tvm.stmt.For):
        for0 = op.first
        
        if isinstance(op.rest, tvm.stmt.Block):
            if isinstance(op.rest.first, tvm.stmt.For):
                for1 = op.rest.first
                # the two loops are the same
                if (for0.loop_var == for1.loop_var
                   and int(for0.min) == int(for1.min)
                   and int(for0.extent) == int(for1.extent)):

                    body = tvm.make.Block(for0.body, for1.body)
                    body = tvm.make.For(for0.loop_var, for0.min, for0.extent, for0.for_type,
                                        for0.device_api, body)
                    body = tvm.make.Block(body, op.rest.rest)
                    return body
    return None

def inject_for_loop(op):
    global src_idx
    global tar_idx

    if op in threads:
        if isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == "thread_extent" and str(op.node.var) == "threadIdx.x":
                src_idx = op.node.var
                tar_idx = src_idx*(max_thds//num_thds) + li

                body = tvm.ir_pass.IRTraverse(op.body, None, inject_for_on_leaf_stmt)
                stmt = tvm.make.AttrStmt(op.node, op.attr_key, tvm.const(num_thds, "int32"), body)
                return stmt
    return None

def thread_loop(stmt):
    global threads
    global head_stmt
    global tail_stmt

    tvm.ir_pass.PostOrderVisit(stmt, find_thread_loop_point)

    if not threads:
        return stmt

    stmt = tvm.ir_pass.IRTransform(stmt, None, inject_for_loop, ['AttrStmt'])
    print("Inject for loops")
    print(stmt)

    stmt = tvm.ir_pass.IRTransform(stmt, None, merge_for_loops, ['Block'])
    print("Merge for loops")
    print(stmt)
    #exit()
    #stmt = tvm.ir_pass.IRTransform(stmt, None, loopify4, ['AttrStmt'])

    return stmt

print("Transformed IR")
#print(thread_loop(ir))

with tvm.build_config(add_lower_pass=[(1, thread_loop)]) as cfg:
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    f = tvm.lower(s, [A, B, C], simple_mode=False)
#exit()

#f = tvm.lower(s, [A, B, C], simple_mode=False)
tgt = 'cuda'

fadd = tvm.build(f, target=tgt)
print("Generated CUDA Code")
print(fadd.imported_modules[0].get_source())
exit()

ctx = tvm.context(tgt, 0)

n = 128
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

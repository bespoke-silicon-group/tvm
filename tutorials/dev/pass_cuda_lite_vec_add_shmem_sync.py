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
def gpu_vec_add_ir_shmem(A, B, C):
    max_threads = 32
    ib = tvm.ir_builder.create()

    with ib.new_scope():
        AA = ib.allocate("float32", 32, name="AA", scope="local")
        BB = ib.allocate("float32", 32, name="BB", scope="local")
        CC = ib.allocate("float32", 32, name="CC", scope="local")

        bx = tvm.thread_axis("blockIdx.x")
        tx = tvm.thread_axis("threadIdx.x")

        ib.scope_attr(bx, "thread_extent", (n+max_threads-1)//max_threads)
        ib.scope_attr(tx, "thread_extent", max_threads)

        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)

        idx = bx.var * max_threads + tx.var

        with ib.if_scope(ib.likely(idx<n)):
            AA[tx.var] = Aptr[idx]
            BB[tx.var] = Bptr[idx]
            CC[tx.var] = AA[tx.var] + BB[tx.var]

            Cptr[idx] = CC[tx.var] 

    body = ib.get()
    return body

C = tvm.extern(A.shape, [A, B], lambda inps, outs: gpu_vec_add_ir_shmem(inps[0], inps[1], outs[0]), 
               name="vec_add", dtype="float32")

s = tvm.create_schedule(C.op)
bounds = tvm.schedule.InferBound(s)
stmt = tvm.schedule.ScheduleOps(s, bounds)
ir  = tvm.lower(s, [A, B, C], simple_mode=True)
print("Original IR")
print(ir)
#exit()

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
            r_body = tvm.make.For(li, 0, tx_num // 4, tvm.stmt.For.Serial, 0, body)
        if isinstance(op.body, tvm.stmt.Block):
            blist = tvm.make.stmt_list(op.body)
            r_body = None
            # create expr for sync
            sync_expr = tvm.make.Call(None, 'tvm_storage_sync', 
                                      tvm.convert(['shared']), 
                                      tvm.expr.Call.Intrinsic, None, 0)
            # make it a stmt, and can be put into stmt.body
            sync = tvm.make.Evaluate(sync_expr)

            for b in blist: 
                if isinstance(b, tvm.stmt.Store):
                    if isinstance(b.index, tvm.expr.Var):
                        body = tvm.ir_pass.Substitute(b, {b.index: b.index*(tx_num//4) + li})
                        body = tvm.make.For(li, 0, tx_num // 4, tvm.stmt.For.Serial, 0, body)
                        r_body = body if r_body is None else tvm.make.Block(r_body, body)
                        r_body = tvm.make.Block(r_body, sync)
                    if isinstance(b.index, tvm.expr.Add):
                        body = tvm.ir_pass.Substitute(b, {b.index.b: b.index.b*(tx_num//4) + li})
                        body = tvm.make.For(li, 0, tx_num // 4, tvm.stmt.For.Serial, 0, body)
                        r_body = body if r_body is None else tvm.make.Block(r_body, body)
                        r_body = tvm.make.Block(r_body, sync)

        r_op = tvm.make.AttrStmt(op.node, op.attr_key, op.value, r_body)
        return r_op
    return None

def thread_loop(stmt):
    global threads

    tvm.ir_pass.PostOrderVisit(stmt, find_thread4)
    #exit()

    if not threads:
        return stmt

    stmt = tvm.ir_pass.IRTransform(stmt, None, loopify4, ['AttrStmt'])

    return stmt

print("Transformed IR")
print(thread_loop(ir))
#exit()

with tvm.build_config(add_lower_pass=[(1, thread_loop)]) as cfg:
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    f = tvm.lower(s, [A, B, C], simple_mode=False)

tgt = 'cuda'

fadd = tvm.build(f, target=tgt)
print("Generated Code")
print(fadd.imported_modules[0].get_source())
exit()

ctx = tvm.context(tgt, 0)

n = 128
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

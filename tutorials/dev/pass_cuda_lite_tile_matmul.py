from __future__ import absolute_import, print_function
import tvm
import numpy as np

######################################################################
# We first write a very simple vector add and build it with the default schedule. Then, we use
# our customized lowering pass to manipulate the IR directly instead of using schedule primitives.
#
n = tvm.const(64, "int32")
A = tvm.placeholder((n, n), name="A")
B = tvm.placeholder((n, n), name="B")
scale = 4
max_threads = 4
block_factor = scale*max_threads
dtype = 'float32'

##### Creating vector addition IR for regular GPU computation #####
def gpu_tile_matmul_ir(A, B, C):
    ib = tvm.ir_builder.create()

    by = tvm.thread_axis("blockIdx.y")
    bx = tvm.thread_axis("blockIdx.x")
    ty = tvm.thread_axis("threadIdx.y")
    tx = tvm.thread_axis("threadIdx.x")

    ib.scope_attr(by, "thread_extent", n//block_factor)
    ib.scope_attr(bx, "thread_extent", n//block_factor)
    ib.scope_attr(ty, "thread_extent", max_threads)
    ib.scope_attr(tx, "thread_extent", max_threads)

    CC = ib.allocate("float32", pow(scale*max_threads, 2), name="CC", scope="shared")
    AA = ib.allocate("float32", pow(max_threads, 2)*scale, name="AA", scope="shared")
    BB = ib.allocate("float32", pow(max_threads, 2)*scale, name="BB", scope="shared")

    Aptr = ib.buffer_ptr(A)
    Bptr = ib.buffer_ptr(B)
    Cptr = ib.buffer_ptr(C)

    start_x = bx*(block_factor)
    start_y = by*(block_factor)

    with ib.new_scope():
        with ib.for_range(0, scale, name="ii.c.init") as i:
            with ib.for_range(0, scale, name="jj.c.init") as j:
                i_y = ty*scale + i
                i_x = tx*scale + j
                CC[i_y*scale*max_threads + i_x] = 0.0

        with ib.for_range(0, n//scale, name="k.outer") as k_out:
            ib.emit(tvm.make.Call(None, 'tvm_storage_sync', tvm.convert(['shared']), tvm.expr.Call.Intrinsic, None, 0))
            with ib.for_range(0, scale, name="ax0.inner") as ax0:
                glb_y = start_y + ty*scale + ax0
                glb_x = k_out*scale + tx
                AA[ty*block_factor +  ax0*scale + tx] = Aptr[glb_y*n + glb_x]
                #glb_y = start_y + ty*scale + tx
                #glb_x = k_out*scale + ax0
                #AA[ty*block_factor + tx*(scale) + ax0] = Aptr[glb_y*n + glb_x]
            with ib.for_range(0, scale, name="ax1.inner") as ax1:
                glb_y = k_out*scale + ty
                glb_x = start_x + tx*scale + ax1
                BB[ty*block_factor + tx*scale + ax1] = Bptr[glb_y*n + glb_x]
                #glb_y = k_out*scale + ax1
                #glb_x = start_x + ty*scale + tx
                #BB[ax1*block_factor + ty*scale + tx] = Bptr[glb_y*n + glb_x]

            ib.emit(tvm.make.Call(None, 'tvm_storage_sync', tvm.convert(['shared']), tvm.expr.Call.Intrinsic, None, 0))

            with ib.for_range(0, scale, name="k.inner") as k_in:
                with ib.for_range(0, scale, name="ii.c") as ii:
                    with ib.for_range(0, scale, name="jj.c") as jj:
                        i_y = ty*scale + ii
                        i_x = tx*scale + jj
                        CC[i_y*scale*max_threads + i_x] = CC[i_y*scale*max_threads + i_x] + (AA[ty*block_factor + ii*scale + k_in] * BB[k_in*block_factor + tx*scale + jj])

    with ib.for_range(0, scale, name="ii.inner.inner") as ii:
        with ib.for_range(0, scale, name="jj.inner.inner") as jj:
            glb_y = start_y + ty*scale + ii 
            glb_x = start_x + tx*scale + jj
            i_y = ty*scale + ii
            i_x = tx*scale + jj
            Cptr[glb_y*n + glb_x] = CC[i_y*scale*max_threads + i_x]

    body = ib.get()
    #print(body)
    return body

print("Original IR")
C = tvm.extern(A.shape, [A, B], lambda inps, outs: 
    gpu_tile_matmul_ir(inps[0], inps[1], outs[0]), name="matmul", dtype="float32")

s = tvm.create_schedule(C.op)
#bounds = tvm.schedule.InferBound(s)
#stmt = tvm.schedule.ScheduleOps(s, bounds)
ir  = tvm.lower(s, [A, B, C], simple_mode=True)
print(ir)
with open('tile_matmul_ir.c', 'w') as f:
    f.write(str(ir))
#exit()

tar_threads = 2
loop_points = []
def find_thread_loop_point(op):
    global loop_points

    if isinstance(op, tvm.stmt.AttrStmt):
        if op.attr_key == "thread_extent" and str(op.node.var) == "threadIdx.y":
            if op.value.value > tar_threads:
                loop_points.append(op)
        elif op.attr_key == "thread_extent" and str(op.node.var) == "threadIdx.x":
            if op.value.value > tar_threads:
                loop_points.append(op)

##### IR Transformation for manycore usage #####
tx_num = None
li = tvm.var('ty')
lj = tvm.var('tx')
loop_var = None
thread_var = None

src_idx = tvm.var()
tar_idx = tvm.var()

def substitute_index_op(op):
    if isinstance(op, tvm.expr.Var):
        if op == thread_var:
            #print(op)
            return op*(max_threads//tar_threads) + loop_var

    return None

def substitute_index(op):
    body = tvm.ir_pass.IRTraverse(op, None, substitute_index_op)
    return body

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
            body = tvm.make.For(loop_var, 0, max_threads//tar_threads, tvm.stmt.For.Serial, 0, op)
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
                    body = tvm.make.For(for0.loop_var, for0.min, for0.extent, 
                                        for0.for_type, for0.device_api, body)
                    body = tvm.make.Block(body, op.rest.rest)
                    return body
    return None

def inject_for_loop(op):
    global src_idx
    global tar_idx
    global loop_var
    global thread_var

    if isinstance(op, tvm.stmt.AttrStmt):
        #print(op.node)
        if op.attr_key == "thread_extent" and (str(op.node.var) == "threadIdx.x" or str(op.node.var) == "threadIdx.y"):
            if op.value.value > tar_threads:
                src_idx = op.node.var
                if op.node.var.name == "threadIdx.y":
                    loop_var = li
                    thread_var = op.node.var
                    tar_idx = src_idx*(max_threads//tar_threads) + li
                elif str(op.node.var) == "threadIdx.x":
                    loop_var = lj
                    thread_var = op.node.var
                    tar_idx = src_idx*(max_threads//tar_threads) + lj

                body = tvm.ir_pass.IRTraverse(op.body, None, inject_for_on_leaf_stmt)
                stmt = tvm.make.AttrStmt(op.node, op.attr_key, tvm.const(tar_threads, "int32"), body)
                return stmt
    return None

def thread_loop(stmt):
    global loop_points
    global head_stmt
    global tail_stmt

    tvm.ir_pass.PostOrderVisit(stmt, find_thread_loop_point)

    if not loop_points:
        return stmt

    stmt = tvm.ir_pass.IRTransform(stmt, None, inject_for_loop, ['AttrStmt'])
    print("Inject for loops")
    #print(stmt)
    #exit()

    #stmt = tvm.ir_pass.IRTransform(stmt, None, merge_for_loops, ['Block'])
    #print("Merge for loops")
    #print(stmt)
    #exit()
    #stmt = tvm.ir_pass.IRTransform(stmt, None, loopify4, ['AttrStmt'])

    return stmt

print("Transformed IR")
#print(thread_loop(ir))

with tvm.build_config(add_lower_pass=[(1, thread_loop)]) as cfg:
    t_ir = tvm.lower(s, [A, B, C], simple_mode=True)
    print(t_ir)
    with open('tr_tile_matmul_ir.c', 'w') as f:
        f.write(str(t_ir))
    f = tvm.lower(s, [A, B, C], simple_mode=False)
#exit()

#f = tvm.lower(s, [A, B, C], simple_mode=False)
tgt = 'cuda'

fmatmul = tvm.build(f, target=tgt, name='matmul')
print("Generated CUDA Code")
code = fmatmul.imported_modules[0].get_source()
print(code)
with open('tile_matmul.cu', 'w') as f:
    f.write(code)
#exit()

ctx = tvm.context(tgt, 0)

n = 64
a_np = np.random.rand(n, n).astype(dtype)
b_np = np.random.rand(n, n).astype(dtype)
c_np = np.zeros((n, n), dtype=dtype)

a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(c_np, ctx)
fmatmul(a, b, c)
ans = np.dot(a.asnumpy(), b.asnumpy())
#print(ans)
err = tvm.testing.assert_allclose(c.asnumpy(), ans, rtol=1e-5)
if not err:
    print("The resutls of matrix multiplication are correct.")

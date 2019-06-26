"""Additional IR Pass for CUDA Lite and HammerBlade"""
from __future__ import absolute_import, print_function
import tvm

# merge_for_loops is not used in the matmul case
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

def inject_thread_loop(stmt):
    ori_threads = None
    tar_threads = 2

    loop_var = None
    thread_var = None

    src_idx = tvm.var()
    tar_idx = tvm.var()

    def substitute_index_op(op):
        if isinstance(op, tvm.expr.Var):
            if op == thread_var:
                return op*(ori_threads//tar_threads) + loop_var
        return None

    def substitute_index(op):
        body = tvm.ir_pass.IRTraverse(op, None, substitute_index_op)
        return body

    def inject_for_on_leaf_stmt(op):
        if isinstance(op, tvm.stmt.Stmt):
            # the Evaluate and Store are the only two types of leaf stmt nodes
            if isinstance(op, tvm.stmt.Store):
                body = tvm.make.For(loop_var, 0, ori_threads//tar_threads, tvm.stmt.For.Serial, 0, op)
                body = tvm.ir_pass.IRTransform(body, None, substitute_index, ['Store'])
                return body
        return None


    def inject_for_loop(op):
        nonlocal ori_threads
        nonlocal loop_var
        nonlocal tar_idx

        if isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == "thread_extent" and \
               (str(op.node.var) == "threadIdx.x" or str(op.node.var) == "threadIdx.y"):
                if op.value.value > tar_threads:
                    ori_threads = op.value.value
                    src_idx = op.node.var
                    if op.node.var.name == "threadIdx.y":
                        loop_var = tvm.var('iy')
                        thread_var = op.node.var
                        tar_idx = src_idx*(ori_threads//tar_threads) + loop_var
                    elif str(op.node.var) == "threadIdx.x":
                        loop_var = tvm.var('ix')
                        thread_var = op.node.var
                        tar_idx = src_idx*(ori_threads//tar_threads) + loop_var

                    body = tvm.ir_pass.IRTraverse(op.body, None, inject_for_on_leaf_stmt)
                    stmt = tvm.make.AttrStmt(op.node, op.attr_key, \
                                             tvm.const(tar_threads, "int32"), body)
                    return stmt
        return None

    stmt = tvm.ir_pass.IRTransform(stmt, None, inject_for_loop, ['AttrStmt'])
    return stmt

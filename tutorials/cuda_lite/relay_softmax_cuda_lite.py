import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
import sys

dtype="float32"
batch_size = 1
input_shape = 16
data_shape = (batch_size, input_shape)
out_shape = (batch_size, input_shape)

def print_stmt(stmt):
    print(stmt)
    return stmt

net, params = relay.testing.softmax.get_workload(
        batch_size=batch_size, 
        input_shape=input_shape,
        dtype=dtype)

# set show_meta_data=True if you want to show meta data
print(net.astext(show_meta_data=False))

data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

opt_level = 0
target = tvm.target.cuda_lite()
with relay.build_config(opt_level=opt_level):
    with tvm.build_config(add_lower_pass=[(1, print_stmt)]):
        graph, lib, _params = relay.build_module.build(
            net, target, params=params)

        ctx = tvm.context("cuda_lite", 0)
        # create module
        module = graph_runtime.create(graph, lib, ctx)
        # set input and parameters
        module.set_input("data", data)
        module.set_input(**params)
        # run
        module.run()
        # get output
        out_cuda_lite = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

target = "llvm"
with relay.build_config(opt_level=opt_level):
    graph, lib, _params = relay.build_module.build(net, target, params=params)

    ctx = tvm.context(target, 0)
    # create module
    module = graph_runtime.create(graph, lib, ctx)
    # set input and parameters
    module.set_input("data", data.astype("float32"))
    module.set_input(**_params)
    # run
    module.run()
    # get output
    out_cpu = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

np.set_printoptions(threshold=sys.maxsize)
print("CUDA-Lite Outputs:")
print(out_cuda_lite.flatten())

print("CPU Outputs:")
print(out_cpu.flatten())

print(np.max(out_cuda_lite.flatten()), np.min(out_cuda_lite.flatten()))
print(np.max(out_cpu.flatten()), np.min(out_cpu.flatten()))

if not tvm.testing.assert_allclose(out_cuda_lite.flatten(), out_cpu.flatten(), rtol=1e-6, atol=1e-3):
    print("CUDA-Lite RESULTS MATCH CPU RESULTS (WITHIN TOLERANCE)")

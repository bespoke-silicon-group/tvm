import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
from hb import ir_pass
import time

dtype="float"
network_scale = 0
batch_size = 1
num_class = 2
image_shape = (3, 8, 8)
data_shape = (batch_size,) + image_shape
#data_shape = (batch_size, num_class)
out_shape = (batch_size, num_class)
#out_shape = (batch_size, ) + (1, 4, 4)

def print_stmt(stmt):
    print(stmt)

    return stmt

net, params = relay.testing.sdh_convnet.get_workload(
              batch_size=batch_size,
              num_classes=num_class,
              image_shape=image_shape,
              scale=network_scale,
              dtype=dtype)

# set show_meta_data=True if you want to show meta data
print(net.astext(show_meta_data=False))
#exit()

data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

opt_level = 3
target = tvm.target.cuda_lite()
cuda_lite_start = time.time()
with relay.build_config(opt_level=opt_level):
    #with tvm.build_config(add_lower_pass=[(1, ir_pass.inject_thread_loop)]):
    with tvm.build_config(add_lower_pass=[(1, print_stmt)]):
        graph, lib, _params = relay.build_module.build(
                net, target, params=params)

        # create random input
        ctx = tvm.context("cuda_lite", 0)
        # create module
        module = graph_runtime.create(graph, lib, ctx)
        # set input and parameters
        module.set_input("data", data)
        module.set_input(**_params)
        # run
        module.run()
        # get output
        out_cuda_lite = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

cuda_lite_end = time.time()
#exit()

target = "llvm"
cpu_start = time.time()
with relay.build_config(opt_level=opt_level):
    graph, lib, _params = relay.build_module.build(net, target, params=params)

    ctx = tvm.context(target, 0)
    # create module
    module = graph_runtime.create(graph, lib, ctx)
    # set input and parameters
    module.set_input("data", data)
    module.set_input(**_params)
    # run
    module.run()
    # get output
    out_cpu = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

cpu_end = time.time()

print("CUDA-Lite Outputs:")
print(out_cuda_lite.flatten())
print("CUDA-Lite Build + Run time: %f" % (cuda_lite_end - cuda_lite_start))

print("CPU Outputs:")
print(out_cpu.flatten())
print("CPU Build + Run time: %f" % (cpu_end - cpu_start))

if not tvm.testing.assert_allclose(out_cuda_lite, out_cpu, rtol=1e-3, atol=1e-2):
    print("CUDA-Lite RESULTS MATCH CPU RESULTS (WITHIN TOLERANCE)")

exit()
######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files and load them
# back in deploy environment.

####################################################

# save the graph, lib and params into separate files
from tvm.contrib import util

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph)
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(relay.save_param_dict(params))
print(temp.listdir())

####################################################
ctx = tvm.context("cuda_lite", 0)

# load the module back.
loaded_json = open(temp.relpath("deploy_graph.json")).read()
loaded_lib = tvm.module.load(path_lib)
loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.run(data=input_data)
out_deploy = module.get_output(0).asnumpy()

# Print first 10 elements of output
print(out_deploy.flatten())

# check whether the output from deployed module is consistent with original one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-3)

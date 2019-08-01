import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
from hb import ir_pass

dtype="float"
batch_size = 1
num_channel = 16
image_shape = (64, 32, 32)
data_shape = (batch_size, ) + image_shape
#out_shape = (batch_size, num_classes)
out_shape = (batch_size, ) + (num_channel, ) + image_shape[1:]

net, params = relay.testing.conv2d.get_workload(
        batch_size=batch_size,
        channels=num_channel,
        image_shape=image_shape,
        dtype=dtype)

# set show_meta_data=True if you want to show meta data
print(net.astext(show_meta_data=False))

opt_level = 3
target = tvm.target.cuda()
#target = "llvm"
with relay.build_config(opt_level=opt_level):
    with relay.build_config():
        with tvm.build_config(add_lower_pass=[(1, ir_pass.inject_thread_loop)]):
            graph, lib, params = relay.build_module.build(
                net, target, params=params)
#exit()

#####################################################################
# Run the generate library
# ------------------------
# Now we can create graph runtime and run the module on Nvidia GPU.

# create random input
ctx = tvm.context("cuda", 0)
#ctx = tvm.context("llvm", 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_runtime.create(graph, lib, ctx)
# set input and parameters
module.set_input("data", data)
module.set_input(**params)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

# Print first 10 elements of output
print(out.flatten())
#print(data)
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
print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with original one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-3)

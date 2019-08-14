import onnx
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
from hb import ir_pass
import time

dtype="float"
network_scale = 1
batch_size = 1
num_class = 2
image_shape = (3, 8, 8)
data_shape = (batch_size,) + image_shape
#data_shape = (batch_size, num_class)
out_shape = (batch_size, num_class)
#out_shape = (batch_size, ) + (1, 4, 4)

model_path = "/home/centos/sdh/convnet/CIFAR2Net_small.onnx"
onnx_model = onnx.load(model_path)

input_name = 'input.1'
shape_dict = {input_name: (1, 3, 8, 8)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print(mod.astext(show_meta_data=False))
#exit()

data = np.random.uniform(-1, 1, size=(1, 3, 8, 8)).astype("float32")

# run on manycore
with relay.build_config(opt_level=3):
    ctx = tvm.context("cuda_lite", 0)
    target = tvm.target.cuda_lite()

    intrp = relay.build_module.create_executor('graph', mod, ctx, target)
    cuda_lite_output = intrp.evaluate()(tvm.nd.array(data), **params).asnumpy()
    print("CUDA-Lite Outputs:")
    print(cuda_lite_output.flatten())

# run on cpu
with relay.build_config(opt_level=3):
    ctx = tvm.cpu(0)
    target = 'llvm'

    intrp = relay.build_module.create_executor('graph', mod, ctx, target)
    cpu_output = intrp.evaluate()(tvm.nd.array(data), **params).asnumpy()
    print("CPU Outputs:")
    print(cpu_output.flatten())

# compare the results from cpu and manycore
if not tvm.testing.assert_allclose(cuda_lite_output, cpu_output, rtol=1e-3, atol=1e-2):
    print("CUDA-Lite RESULTS MATCH CPU RESULTS (WITHIN TOLERANCE)")

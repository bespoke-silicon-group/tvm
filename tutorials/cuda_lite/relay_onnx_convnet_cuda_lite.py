import onnx
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from PIL import Image
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

#model_path = "/home/centos/sdh/convnet/conv.onnx"
model_path = "./conv.onnx"
onnx_model = onnx.load(model_path)

input_name = 'input.1'
shape_dict = {input_name: (1, 3, 8, 8)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print(mod.astext(show_meta_data=False))
#exit()

#data = np.random.uniform(-1, 1, size=(1, 3, 8, 8)).astype("float32")
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((8, 8))
#img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
#img_y, img_cb, img_cr = img_ycbcr.split()
#x = np.array(img_y)[np.newaxis, np.newaxis, :, :]
#x = x/255
x = np.array(img)
x = np.moveaxis(x, 2, 0)
x = x/255
x = np.array(x)[np.newaxis, :, :, :]

# run on manycore
with relay.build_config(opt_level=3):
    ctx = tvm.context("cuda_lite", 0)
    target = tvm.target.cuda_lite()

    intrp = relay.build_module.create_executor('graph', mod, ctx, target)
    cuda_lite_output = intrp.evaluate()(tvm.nd.array(x.astype("float32")), **params).asnumpy()
#exit()

# run on cpu
with relay.build_config(opt_level=3):
    ctx = tvm.cpu(0)
    target = 'llvm'

    intrp = relay.build_module.create_executor('graph', mod, ctx, target)
    cpu_output = intrp.evaluate()(tvm.nd.array(x.astype("float32")), **params).asnumpy()

print("CUDA-Lite Outputs:")
print(cuda_lite_output.flatten())

print("CPU Outputs:")
print(cpu_output.flatten())
# compare the results from cpu and manycore
if not tvm.testing.assert_allclose(cuda_lite_output, cpu_output, rtol=1e-3, atol=1e-2):
    print("CUDA-Lite RESULTS MATCH CPU RESULTS (WITHIN TOLERANCE)")

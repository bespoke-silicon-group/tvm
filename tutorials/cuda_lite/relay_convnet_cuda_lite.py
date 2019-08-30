import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from PIL import Image
import time
import sys

dtype="float32"
network_scale = 0
batch_size = 1
num_class = 2
image_shape = (3, 16, 16)
data_shape = (batch_size,) + image_shape
#data_shape = (batch_size, num_class)
out_shape = (batch_size, num_class)
#out_shape = (batch_size, ) + (32, 16, 16)

def print_stmt(stmt):
    print(stmt)
    return stmt

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((16, 16))
x = np.array(img)
x = np.moveaxis(x, 2, 0)
x = np.array(x)[np.newaxis, :, :, :]
x = x/255
#img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
#img_y, img_cb, img_cr = img_ycbcr.split()
#x = np.array(img_y)[np.newaxis, np.newaxis, :, :]
#x = x/255

net, params = relay.testing.sdh_convnet.get_workload(
              batch_size=batch_size,
              num_classes=num_class,
              image_shape=image_shape,
              scale=network_scale,
              dtype=dtype)

# set show_meta_data=True if you want to show meta data
print(net.astext(show_meta_data=False))
#exit()

opt_level = 3
target = tvm.target.cuda_lite()
with relay.build_config(opt_level=opt_level):
    with tvm.build_config(add_lower_pass=[(1, print_stmt)]):
        graph, lib, _params = relay.build_module.build(
                net, target, params=params)

        # create random input
        ctx = tvm.context("cuda_lite", 0)
        # create module
        module = graph_runtime.create(graph, lib, ctx)
        # set input and parameters
        module.set_input("data", x.astype("float32"))
        module.set_input(**_params)
        # run
        cuda_lite_start = time.time()
        module.run()
        cuda_lite_end = time.time()
        # get output
        out_cuda_lite = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()


target = "llvm"
with relay.build_config(opt_level=opt_level):
    graph, lib, _params = relay.build_module.build(net, target, params=params)

    ctx = tvm.context(target, 0)
    # create module
    module = graph_runtime.create(graph, lib, ctx)
    # set input and parameters
    module.set_input("data", x.astype("float32"))
    module.set_input(**_params)
    # run
    cpu_start = time.time()
    module.run()
    cpu_end = time.time()
    # get output
    out_cpu = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()


np.set_printoptions(threshold=sys.maxsize)
print("CUDA-Lite Outputs:")
print(out_cuda_lite.flatten())
print("CUDA-Lite Build + Run time: %f" % (cuda_lite_end - cuda_lite_start))

print("CPU Outputs:")
print(out_cpu.flatten())
print("CPU Build + Run time: %f" % (cpu_end - cpu_start))

if not tvm.testing.assert_allclose(out_cuda_lite.flatten(), out_cpu.flatten(), rtol=1e-3, atol=1e-2):
    print("CUDA-Lite RESULTS MATCH CPU RESULTS (WITHIN TOLERANCE)")

exit()

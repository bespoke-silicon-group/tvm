# pylint: disable=unused-argument
from tvm import relay
from .init import create_workload
from . import layers

def block_unit(data,
               num_filter,
               stride,
               name,
               residual=True):

    conv1 = layers.conv2d(
            data=data, channels=num_filter, kernel_size=(3, 3),
            strides=stride, padding=(1, 1), name=name + '_conv1')
    #bn1 = layers.batch_norm_infer(data=conv1, epsilon=1e-5, name=name + '_bn1')
    #act1 = relay.nn.relu(data=bn1)
    act1 = relay.nn.relu(data=conv1)
    #act1 = conv1

    if residual:
        shortcut = layers.conv2d(
            data=data, channels=num_filter, kernel_size=(1, 1),
            strides=stride, padding=(0, 0), name=name + '_sc')
        return relay.add(act1, shortcut)
    else:
        return act1


def sdh_convnet(units,
                num_stages,
                filter_list,
                num_classes,
                data_shape,
                dtype="float32"):

    num_unit = len(units)
    assert num_unit == num_stages
    data = relay.var("data", shape=data_shape, dtype=dtype)

    body = block_unit(data, filter_list[0], (1, 1), name='block0', residual=False)
    body = block_unit(body, filter_list[1], (1, 1), name='block1', residual=True)
    body = relay.nn.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2))
    body = block_unit(body, filter_list[2], (1, 1), name='block2', residual=True)
    body = relay.nn.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2))
    body = block_unit(body, filter_list[3], (1, 1), name='block3', residual=True)
    
    pool1 = relay.nn.global_max_pool2d(data=body)
    flat = relay.nn.batch_flatten(data=pool1)
    fc1 = relay.nn.dense(flat, relay.var("fc1_weight"), units=num_classes)
    net = relay.nn.softmax(data=fc1)

    args = relay.ir_pass.free_vars(net)
    return relay.Function(args, net)


def get_net(batch_size,
            num_classes,
            num_layers=4,
            image_shape=(3, 32, 32),
            scale=0,
            dtype="float32",
            **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    (_, height, _) = image_shape
    data_shape = (batch_size,) + image_shape

    num_stages = 4
    units = [1, 1, 1, 1]

    if scale == 0:
        filter_list = [4, 8, 16, 32]
    elif scale == 1:
        filter_list = [8, 16, 32, 64]
    elif scale == 2:
        filter_list = [16, 32, 64, 128]

    return sdh_convnet(units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  num_classes=num_classes,
                  data_shape=data_shape,
                  dtype=dtype)


def get_workload(batch_size=1,
                 num_classes=2,
                 num_layers=4,
                 image_shape=(3, 32, 32),
                 dtype="float32",
                 **kwargs):

    net = get_net(batch_size=batch_size,
                  num_classes=num_classes,
                  num_layers=num_layers,
                  image_shape=image_shape,
                  dtype=dtype,
                  **kwargs)
    return create_workload(net)

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
a simple multilayer perceptron
"""
from __future__ import absolute_import
from tvm import relay
from .init import create_workload

def get_net(batch_size=1,
            num_neurons=16,
            input_shape=(1, 8, 8),
            dtype="float32"):

    #data_shape = (batch_size, ) + input_shape
    data_shape = (batch_size, input_shape)
    data = relay.var("data",
                     shape=data_shape,
                     dtype=dtype)
    #data = relay.nn.batch_flatten(data)
    fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=num_neurons)
    args = relay.ir_pass.free_vars(fc1)
    return relay.Function(args, fc1)


def get_workload(batch_size,
                 num_neurons=16,
                 input_shape=(1, 8, 8),
                 dtype="float32"):

    net = get_net(batch_size, num_neurons, input_shape, dtype)
    return create_workload(net)

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
"""Verilator codegen tests"""

import numpy as np

import tvm
from tvm import relay

from test_verilator.infrastructure import (
    _register_verilator_op,
    skip_test,
    compile_module,
    run_module,
    offload,
    clear_stats,
    stats,
)

_register_verilator_op("add")
_register_verilator_op("nn.bias_add")


def create_module_add(shape, dtype):
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    z = relay.add(x, y)
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def create_module_bias_add(xshape, yshape, dtype):
    x = relay.var("x", shape=xshape, dtype=dtype)
    y = relay.var("y", shape=yshape, dtype=dtype)
    z = relay.nn.bias_add(x, y, axis=3)
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def run_and_check(exe, xshape, yshape, dtype):
    x_data = np.random.randint(5, size=xshape, dtype=dtype)
    y_data = np.random.randint(5, size=yshape, dtype=dtype)
    ref = x_data + y_data
    inputs = {"x": x_data, "y": y_data}
    clear_stats()
    out = run_module(exe, inputs)
    values = stats()
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)
    return values["cycle_counter"]


def print_counter(test, lanes, cycles):
    print(
        "test:{} vector-lanes:{} number of cycles:{}".format(
            test, lanes, cycles
        )
    )


def tadd(lanes):
    if skip_test():
        return
    dtype = "int32"
    shape = (8, 4)
    mod = create_module_add(shape, dtype)
    mod = offload(mod)
    exe = compile_module(mod, lanes)
    cycles = run_and_check(exe, shape, shape, dtype)
    print_counter("add", lanes, cycles)


def tbias(lanes):
    if skip_test():
        return
    dtype = "int32"
    xshape = (1, 112, 112, 32)
    yshape = (32,)
    mod = create_module_bias_add(xshape, yshape, dtype)
    mod = offload(mod)
    exe = compile_module(mod, lanes)
    cycles = run_and_check(exe, xshape, yshape, dtype)
    print_counter("nn.bias_add", lanes, cycles)


def test_adds():
    print("\nTesting multiple vector lanes on different operators...")
    tadd(1)
    tadd(2)
    tbias(1)
    tbias(32)

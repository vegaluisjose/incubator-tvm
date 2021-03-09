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
"""Verilator utility functions"""

import os
import sys
import subprocess as sp
import json

import tvm
from tvm import relay
import tvm.relay.testing
from tvm import runtime
from tvm.relay import transform


def _register_verilator_op(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported by Verilator.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.verilator")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


def skip_test():
    """Skip test if it requires the Verilator codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.verilator", True):
        print("Skip test because Verilator codegen is not available.")
        return True
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return True
    return False


def clear_stats():
    """Clear profiler statistics."""
    f = tvm.get_global_func("verilator.profiler_clear", True)
    if f:
        f()


def stats():
    """Get profiler statistics."""

    x = tvm.get_global_func("verilator.profiler_status")()
    return json.loads(x)


def offload(mod):
    """Offload ops based on the registered ops."""

    backend = "verilator"
    mod = transform.AnnotateTarget([backend])(mod)
    mod = transform.PartitionGraph()(mod)
    return mod


def verilator_app_path():
    """Find verilator hardware app path."""

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        cur_dir,
        "..",
        "..",
        "..",
        "..",
        "3rdparty",
        "vta-hw",
        "apps",
        "verilator",
        "add",
    )


def compile_hardware(lib_name, lanes):
    """Compile hardware into shared library

    Paramters
    ---------
    name : Str
        The name of the library.

    lanes : Int
        The number of vector lanes.
    """

    opt_lib_name = "LIB_NAME={}".format(lib_name)
    opt_lanes = "LANES={}".format(lanes)

    cmd = []
    cmd.append("make")
    cmd.append("--directory")
    cmd.append(verilator_app_path())
    cmd.append(opt_lib_name)
    cmd.append(opt_lanes)
    sp.run(cmd, check=True)


def compile_module(mod, lanes):
    """Compile Relay module and hardware library

    Paramters
    ---------
    mod : Module
        The Relay Module.

    lanes : Int
        The number of vector lanes.
    """

    lib_name = "libverilator_{}".format(lanes)
    lib_name_ext = "{}.so".format(lib_name)
    lib = os.path.join(verilator_app_path(), lib_name_ext)
    if not os.path.isfile(lib):
        compile_hardware(lib_name, lanes)

    opts = {
        "lib_path": lib,
        "profiler_enable": True,
        "profiler_cycle_counter_id": 0,
    }

    with tvm.transform.PassContext(
        opt_level=3, config={"relay.ext.verilator.options": opts}
    ):
        exe = relay.vm.compile(mod, target="llvm", params=None)
        code, lib = exe.save()
        return runtime.vm.Executable.load_exec(code, lib)


def run_module(exe, inputs):
    """Run Relay module"""

    ctx = tvm.cpu()
    vm = runtime.vm.VirtualMachine(exe, ctx)
    return vm.run(**inputs)

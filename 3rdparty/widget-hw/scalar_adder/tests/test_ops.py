# this is temporary just for testing verilator backend
import os

import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
from tvm.contrib import util
from tvm import runtime
from tvm.relay import transform


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.verilator")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper


_register_external_op_helper("add")


def run_prog(exe, inputs):
    ctx = tvm.cpu()
    vm = runtime.vm.VirtualMachine(exe, ctx)
    return vm.run(**inputs)


def compile_prog(mod):
    with relay.build_config(opt_level=3):
        exe = relay.vm.compile(mod, target="llvm", params=None)
        code, lib = exe.save()
        return runtime.vm.Executable.load_exec(code, lib)


def partition_prog(mod, backend):
    mod = transform.AnnotateTarget([backend])(mod)
    mod = transform.PartitionGraph()(mod)
    return mod


def build_add(shape, dtype):
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    z = relay.add(x, y)
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def run_add(exe, shape, dtype):
    x_data = np.random.randint(5, size=shape, dtype=dtype)
    y_data = np.random.randint(5, size=shape, dtype=dtype)
    ref = x_data + y_data
    inputs = {"x": x_data, "y": y_data}
    out = run_prog(exe, inputs)
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def test_add(backend):
    dtype = "int32"
    shape = (8, 4)
    mod = build_add(shape, dtype)
    mod = partition_prog(mod, backend)
    exe = compile_prog(mod)
    run_add(exe, shape, dtype)


if __name__ == "__main__":
    backend = "verilator"
    test_add(backend)

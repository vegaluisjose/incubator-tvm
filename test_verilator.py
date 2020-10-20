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


def run(exe, inputs):
    ctx = tvm.cpu()
    vm = runtime.vm.VirtualMachine(exe, ctx)
    return vm.run(**inputs)


def update_lib(lib):
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    contrib_path = os.path.join(test_dir, "..", "src", "runtime", "contrib")

    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    tmp_path = util.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = runtime.load_module(lib_path)

    return lib


def compile_prog(mod, params=None):
    with relay.build_config(opt_level=3):
        exe = relay.vm.compile(mod, target="llvm", params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        return runtime.vm.Executable.load_exec(code, lib)


def build_add_program(shape, dtype):
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
    print("x:\n", x_data)
    print("y:\n", y_data)
    out = run(exe, inputs)
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def partition(mod, backend):
    mod = transform.AnnotateTarget([backend])(mod)
    print(mod)
    mod = transform.PartitionGraph()(mod)
    return mod


def test_add(backend):
    dtype = "int32"
    shape = (8, 8)
    mod = build_add_program(shape, dtype)
    print(mod)
    mod = partition(mod, backend)
    print(mod)
    exe = compile_prog(mod)
    run_add(exe, shape, dtype)


if __name__ == "__main__":
    backend = "verilator"
    test_add(backend)

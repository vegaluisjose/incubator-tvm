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

import tvm
from tvm import te, relay, transform
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime as runtime

import os
from PIL import Image
import numpy as np

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


_register_verilator_op("nn.bias_add")

def offload(mod):
    """Offload ops based on the registered ops."""

    backend = "verilator"
    mod = tvm.relay.transform.AnnotateTarget([backend])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    return mod


def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


def get_mobilenet():
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
    model_path = download_testdata(model_url, "mobilenet_v1_1.0_224_quant.tgz", module=["tf", "official"])
    model_dir = os.path.dirname(model_path)
    extract(model_path)
    tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224_quant.tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()
    try:
        import tflite
    
        return tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
    
        return tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

def get_real_image(im_height, im_width):
    repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'
    img_name = 'elephant-299.jpg'
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module='data')
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype('uint8')
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data

# TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "uint8"

tflite_model = get_mobilenet()

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

mod = offload(mod)

opts = {
    "lib_path": "/home/vega/github/tvm/3rdparty/vta-hw/apps/verilator/add/libverilator.so",
    "profiler_enable": True,
    "profiler_cycle_counter_id": 0,
}

# Build the module against to x86 CPU
target = "llvm"
with transform.PassContext(opt_level=3, config={"relay.ext.verilator.options": opts}):
    lib = relay.build(mod, target, params=params)


module = runtime.GraphModule(lib["default"](tvm.cpu()))
image_data = get_real_image(224, 224)
module.set_input(input_tensor, image_data)
module.run()
tvm_output = module.get_output(0).asnumpy()

label_file_url = "".join(
    [
        "https://raw.githubusercontent.com/",
        "tensorflow/tensorflow/master/tensorflow/lite/java/demo/",
        "app/src/main/assets/",
        "labels_mobilenet_quant_v1_224.txt",
    ]
)
label_file = "labels_mobilenet_quant_v1_224.txt"
label_path = download_testdata(label_file_url, label_file, module="data")

# List of 1001 classes
with open(label_path) as f:
    labels = f.readlines()

# Convert result to 1D data
predictions = np.squeeze(tvm_output)

# Get top 1 prediction
prediction = np.argmax(predictions)

# Convert id to class name and show the result
print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])

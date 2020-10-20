/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/verilator/verilator_runtime.cc
 * \brief A simple JSON runtime for Verilator.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class VerilatorJSONRuntime : public JSONRuntimeBase {
 public:
  VerilatorJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                       const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "verilator_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      uint32_t eid = EntryID(nid, 0);
      if (nodes_[nid].GetOpType() == "input") {
        int* data = static_cast<int*>(data_entry_[eid]->data);
        std::cout << data[0] << std::endl;
      }
    }
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("add" == op_name) {
          Add(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  void Add(const size_t& nid) {
    std::cout << "Running Add" << std::endl;
  }
};

runtime::Module VerilatorJSONRuntimeCreate(String symbol_name, String graph_json,
                                           const Array<String>& const_names) {
  auto n = make_object<VerilatorJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.VerilatorJSONRuntimeCreate")
    .set_body_typed(VerilatorJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_verilator_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<VerilatorJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

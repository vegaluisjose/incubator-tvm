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
 * \file src/runtime/contrib/verilator/verilator.cc
 * \brief TVM compatible wrappers for verilator kernels.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace verilator {

extern "C" void add(int* data, int* weight, int* out, int h, int w) {
  for (int64_t i = 0; i < h; ++i) {
    for (int64_t j = 0; j < w; ++j) {
      int64_t k = i * w + j;
      out[k] = data[k] + weight[k];
    }
  }
}

}  // namespace verilator
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

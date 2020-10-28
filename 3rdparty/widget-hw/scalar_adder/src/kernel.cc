#include <stdint.h>

#include "verilator_device.h"
#include "verilator_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

// void verilator_add(VerilatorHandle handle, int *a, int *b, int *y, int h, int w) {
//   VerilatorReset(handle, 1);
//   for (int64_t i = 0; i < h; ++i) {
//     for (int64_t j = 0; j < w; ++j) {
//       int64_t k = i * w + j;
//       VerilatorWrite(handle, 0, 0, a[k]);
//       VerilatorWrite(handle, 1, 0, b[k]);
//       VerilatorRun(handle, 1);
//       y[k] = VerilatorRead(handle, 2, 0);
//     }
//   }
// }

void verilator_add(VerilatorHandle handle, int* a, int* b, int* y, int h, int w) {
  for (int64_t i = 0; i < h; ++i) {
    for (int64_t j = 0; j < w; ++j) {
      int64_t k = i * w + j;
      y[k] = a[k] + b[k];
    }
  }
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

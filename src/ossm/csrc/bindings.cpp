#include <torch/extension.h>

namespace ossm {
torch::Tensor linoss_scan(const at::Tensor& m11,
                           const at::Tensor& m12,
                           const at::Tensor& m21,
                           const at::Tensor& m22,
                           const at::Tensor& b_seq);

at::Tensor lru_scan(const at::Tensor& lambda_real,
                    const at::Tensor& lambda_imag,
                    const at::Tensor& b_seq);

at::Tensor linear_rnn_scan(const at::Tensor& weight_hh,
                           const at::Tensor& weight_xh,
                           const at::Tensor& bias,
                           const at::Tensor& inputs,
                           const at::Tensor& initial_state);

at::Tensor s5_scan(const at::Tensor& lambda_real,
                   const at::Tensor& lambda_imag,
                   const at::Tensor& b_seq);
}  // namespace ossm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linoss_scan", &ossm::linoss_scan, "LinOSS associative scan kernel");
  m.def("lru_scan", &ossm::lru_scan, "LRU associative scan kernel");
  m.def("linear_rnn_scan", &ossm::linear_rnn_scan, "Linear RNN scan kernel");
  m.def("s5_scan", &ossm::s5_scan, "S5 associative scan kernel");
}

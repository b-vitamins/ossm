#include <torch/extension.h>

namespace ossm {
torch::Tensor linoss_scan(const at::Tensor& m11,
                           const at::Tensor& m12,
                           const at::Tensor& m21,
                           const at::Tensor& m22,
                           const at::Tensor& b_seq);

torch::Tensor dlinoss_imex1_forward(const at::Tensor& a_diag,
                                    const at::Tensor& g_diag,
                                    const at::Tensor& step,
                                    const at::Tensor& bu);

std::vector<at::Tensor> dlinoss_imex1_backward(const at::Tensor& a_diag,
                                               const at::Tensor& g_diag,
                                               const at::Tensor& step,
                                               const at::Tensor& bu,
                                               const at::Tensor& states,
                                               const at::Tensor& grad_output);

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
  m.def("dlinoss_imex1_forward", &ossm::dlinoss_imex1_forward, "D-LinOSS IMEX1 forward kernel");
  m.def("dlinoss_imex1_backward", &ossm::dlinoss_imex1_backward, "D-LinOSS IMEX1 backward kernel");
  m.def("lru_scan", &ossm::lru_scan, "LRU associative scan kernel");
  m.def("linear_rnn_scan", &ossm::linear_rnn_scan, "Linear RNN scan kernel");
  m.def("s5_scan", &ossm::s5_scan, "S5 associative scan kernel");
}

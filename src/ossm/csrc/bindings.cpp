#include <torch/extension.h>

namespace py = pybind11;

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

torch::Tensor dlinoss_im_forward(const at::Tensor& a_diag,
                                 const at::Tensor& g_diag,
                                 const at::Tensor& step,
                                 const at::Tensor& bu);

std::vector<at::Tensor> dlinoss_im_backward(const at::Tensor& a_diag,
                                            const at::Tensor& g_diag,
                                            const at::Tensor& step,
                                            const at::Tensor& bu,
                                            const at::Tensor& states,
                                            const at::Tensor& grad_output);

torch::Tensor dlinoss_ex_forward(const at::Tensor& a_diag,
                                 const at::Tensor& g_diag,
                                 const at::Tensor& step,
                                 const at::Tensor& bu);

std::vector<at::Tensor> dlinoss_ex_backward(const at::Tensor& a_diag,
                                            const at::Tensor& g_diag,
                                            const at::Tensor& step,
                                            const at::Tensor& bu,
                                            const at::Tensor& states,
                                            const at::Tensor& grad_output);

torch::Tensor dlinoss_imex2_forward(const at::Tensor& a_diag,
                                    const at::Tensor& g_diag,
                                    const at::Tensor& step,
                                    const at::Tensor& bu);

std::vector<at::Tensor> dlinoss_imex2_backward(const at::Tensor& a_diag,
                                               const at::Tensor& g_diag,
                                               const at::Tensor& step,
                                               const at::Tensor& bu,
                                               const at::Tensor& states,
                                               const at::Tensor& grad_output);

torch::Tensor sdlinoss_imex1_forward(const at::Tensor& A,
                                     const at::Tensor& G,
                                     const at::Tensor& step,
                                     const at::Tensor& bu);

std::vector<at::Tensor> sdlinoss_imex1_backward(const at::Tensor& A,
                                                const at::Tensor& G,
                                                const at::Tensor& step,
                                                const at::Tensor& bu,
                                                const at::Tensor& states,
                                                const at::Tensor& grad_output);

torch::Tensor sdlinoss_imex2_forward(const at::Tensor& A,
                                     const at::Tensor& G,
                                     const at::Tensor& step,
                                     const at::Tensor& bu);

std::vector<at::Tensor> sdlinoss_imex2_backward(const at::Tensor& A,
                                                const at::Tensor& G,
                                                const at::Tensor& step,
                                                const at::Tensor& bu,
                                                const at::Tensor& states,
                                                const at::Tensor& grad_output);

torch::Tensor sdlinoss_im_forward(const at::Tensor& A,
                                  const at::Tensor& G,
                                  const at::Tensor& step,
                                  const at::Tensor& bu);

std::vector<at::Tensor> sdlinoss_im_backward(const at::Tensor& A,
                                             const at::Tensor& G,
                                             const at::Tensor& step,
                                             const at::Tensor& bu,
                                             const at::Tensor& states,
                                             const at::Tensor& grad_output);

torch::Tensor sdlinoss_ex_forward(const at::Tensor& A,
                                  const at::Tensor& G,
                                  const at::Tensor& step,
                                  const at::Tensor& bu);

std::vector<at::Tensor> sdlinoss_ex_backward(const at::Tensor& A,
                                             const at::Tensor& G,
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

at::Tensor selective_scan_cpu(const at::Tensor& inputs,
                              const at::Tensor& dt,
                              const at::Tensor& A,
                              const at::Tensor& B,
                              const at::Tensor& C,
                              const c10::optional<at::Tensor>& gate);

std::vector<at::Tensor> selective_scan_cpu_backward(const at::Tensor& grad_output,
                                                    const at::Tensor& inputs,
                                                    const at::Tensor& dt,
                                                    const at::Tensor& A,
                                                    const at::Tensor& B,
                                                    const at::Tensor& C,
                                                    const c10::optional<at::Tensor>& gate);

#ifdef WITH_CUDA
std::vector<at::Tensor> selective_scan_cuda_forward(const at::Tensor& inputs,
                                                    const at::Tensor& dt,
                                                    const at::Tensor& A,
                                                    const at::Tensor& B,
                                                    const at::Tensor& C,
                                                    const c10::optional<at::Tensor>& gate,
                                                    int64_t chunk_length);

std::vector<at::Tensor> selective_scan_cuda_backward(const at::Tensor& grad_output,
                                                     const at::Tensor& inputs,
                                                     const at::Tensor& dt,
                                                     const at::Tensor& A,
                                                     const at::Tensor& B,
                                                     const at::Tensor& C,
                                                     const c10::optional<at::Tensor>& gate,
                                                     const at::Tensor& chunk_states,
                                                     int64_t chunk_length);
#endif
}  // namespace ossm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linoss_scan", &ossm::linoss_scan, "LinOSS associative scan kernel");
  m.def("dlinoss_imex1_forward", &ossm::dlinoss_imex1_forward, "D-LinOSS IMEX1 forward kernel");
  m.def("dlinoss_imex1_backward", &ossm::dlinoss_imex1_backward, "D-LinOSS IMEX1 backward kernel");
  m.def("dlinoss_im_forward", &ossm::dlinoss_im_forward, "D-LinOSS IM forward kernel");
  m.def("dlinoss_im_backward", &ossm::dlinoss_im_backward, "D-LinOSS IM backward kernel");
  m.def("dlinoss_ex_forward", &ossm::dlinoss_ex_forward, "D-LinOSS EX forward kernel");
  m.def("dlinoss_ex_backward", &ossm::dlinoss_ex_backward, "D-LinOSS EX backward kernel");
  m.def("dlinoss_imex2_forward", &ossm::dlinoss_imex2_forward, "D-LinOSS IMEX2 forward kernel");
  m.def("dlinoss_imex2_backward", &ossm::dlinoss_imex2_backward, "D-LinOSS IMEX2 backward kernel");
  m.def("sdlinoss_imex1_forward", &ossm::sdlinoss_imex1_forward, "Selective D-LinOSS IMEX1 forward kernel");
  m.def("sdlinoss_imex1_backward", &ossm::sdlinoss_imex1_backward, "Selective D-LinOSS IMEX1 backward kernel");
  m.def("sdlinoss_imex2_forward", &ossm::sdlinoss_imex2_forward, "Selective D-LinOSS IMEX2 forward kernel");
  m.def("sdlinoss_imex2_backward", &ossm::sdlinoss_imex2_backward, "Selective D-LinOSS IMEX2 backward kernel");
  m.def("sdlinoss_im_forward", &ossm::sdlinoss_im_forward, "Selective D-LinOSS IM forward kernel");
  m.def("sdlinoss_im_backward", &ossm::sdlinoss_im_backward, "Selective D-LinOSS IM backward kernel");
  m.def("sdlinoss_ex_forward", &ossm::sdlinoss_ex_forward, "Selective D-LinOSS EX forward kernel");
  m.def("sdlinoss_ex_backward", &ossm::sdlinoss_ex_backward, "Selective D-LinOSS EX backward kernel");
  m.def("lru_scan", &ossm::lru_scan, "LRU associative scan kernel");
  m.def("linear_rnn_scan", &ossm::linear_rnn_scan, "Linear RNN scan kernel");
  m.def("s5_scan", &ossm::s5_scan, "S5 associative scan kernel");
  m.def("selective_scan",
        &ossm::selective_scan_cpu,
        "Fused selective scan with SiLU gate (CPU)",
        py::arg("inputs"),
        py::arg("dt"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("gate") = py::none());
  m.def("selective_scan_backward",
        &ossm::selective_scan_cpu_backward,
        "Backward pass for fused selective scan (CPU)",
        py::arg("grad_output"),
        py::arg("inputs"),
        py::arg("dt"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("gate") = py::none());
#ifdef WITH_CUDA
  m.def("selective_scan_cuda",
        &ossm::selective_scan_cuda_forward,
        "Fused selective scan with SiLU gate (CUDA)",
        py::arg("inputs"),
        py::arg("dt"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("gate") = py::none(),
        py::arg("chunk_length") = 16);
  m.def("selective_scan_cuda_backward",
        &ossm::selective_scan_cuda_backward,
        "Backward pass for fused selective scan (CUDA)",
        py::arg("grad_output"),
        py::arg("inputs"),
        py::arg("dt"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("gate") = py::none(),
        py::arg("chunk_states"),
        py::arg("chunk_length"));
#endif
}

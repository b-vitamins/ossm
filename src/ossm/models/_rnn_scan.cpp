#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <torch/extension.h>
#include <vector>

namespace {

at::Tensor linear_rnn_scan_cpu(const at::Tensor& weight_hh,
                               const at::Tensor& weight_xh,
                               const at::Tensor& bias,
                               const at::Tensor& inputs,
                               const at::Tensor& initial_state) {
  TORCH_CHECK(weight_hh.device().is_cpu(), "weight_hh must be on CPU");
  TORCH_CHECK(weight_xh.device().is_cpu(), "weight_xh must be on CPU");
  TORCH_CHECK(bias.device().is_cpu(), "bias must be on CPU");
  TORCH_CHECK(inputs.device().is_cpu(), "inputs must be on CPU");
  TORCH_CHECK(initial_state.device().is_cpu(), "initial_state must be on CPU");

  TORCH_CHECK(weight_hh.dim() == 2, "weight_hh must be 2-D");
  TORCH_CHECK(weight_xh.dim() == 2, "weight_xh must be 2-D");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1-D");
  TORCH_CHECK(inputs.dim() == 3, "inputs must be 3-D");
  TORCH_CHECK(initial_state.dim() == 2, "initial_state must be 2-D");

  const auto hidden_size = weight_hh.size(0);
  TORCH_CHECK(weight_hh.size(1) == hidden_size,
              "weight_hh must be square with size hidden_size x hidden_size");
  TORCH_CHECK(weight_xh.size(0) == hidden_size,
              "weight_xh rows must equal hidden_size");
  const auto input_size = weight_xh.size(1);

  TORCH_CHECK(bias.size(0) == hidden_size, "bias must match hidden_size");

  TORCH_CHECK(inputs.size(2) == input_size,
              "inputs last dimension must equal input_size");
  TORCH_CHECK(initial_state.size(1) == hidden_size,
              "initial_state second dimension must equal hidden_size");
  TORCH_CHECK(initial_state.size(0) == inputs.size(1),
              "initial_state batch dimension must match inputs");

  const auto length = inputs.size(0);
  const auto batch = inputs.size(1);

  auto weight_hh_contig = weight_hh.contiguous();
  auto weight_xh_contig = weight_xh.contiguous();
  auto bias_contig = bias.contiguous();
  auto inputs_contig = inputs.contiguous();
  auto initial_contig = initial_state.contiguous();

  auto outputs = at::empty({length, batch, hidden_size}, inputs.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs_contig.scalar_type(),
                                      "linear_rnn_scan_cpu",
                                      [&] {
                                        const auto* weight_hh_ptr =
                                            weight_hh_contig.data_ptr<scalar_t>();
                                        const auto* weight_xh_ptr =
                                            weight_xh_contig.data_ptr<scalar_t>();
                                        const auto* bias_ptr = bias_contig.data_ptr<scalar_t>();
                                        const auto* inputs_ptr = inputs_contig.data_ptr<scalar_t>();
                                        const auto* init_ptr = initial_contig.data_ptr<scalar_t>();
                                        auto* out_ptr = outputs.data_ptr<scalar_t>();

                                        for (int64_t b = 0; b < batch; ++b) {
                                          std::vector<scalar_t> state_vec(hidden_size);
                                          std::vector<scalar_t> next_vec(hidden_size);
                                          std::copy(
                                              init_ptr + b * hidden_size,
                                              init_ptr + (b + 1) * hidden_size,
                                              state_vec.begin());

                                          for (int64_t t = 0; t < length; ++t) {
                                            const scalar_t* input_t =
                                                inputs_ptr + (t * batch + b) * input_size;
                                            scalar_t* out_t = out_ptr + (t * batch + b) * hidden_size;

                                            at::parallel_for(0, hidden_size, 0, [&](int64_t start, int64_t end) {
                                              for (int64_t h = start; h < end; ++h) {
                                                scalar_t acc = bias_ptr[h];
                                                const scalar_t* w_h_row =
                                                    weight_hh_ptr + h * hidden_size;
                                                const scalar_t* w_x_row =
                                                    weight_xh_ptr + h * input_size;
                                                for (int64_t k = 0; k < hidden_size; ++k) {
                                                  acc += w_h_row[k] * state_vec[k];
                                                }
                                                for (int64_t k = 0; k < input_size; ++k) {
                                                  acc += w_x_row[k] * input_t[k];
                                                }
                                                next_vec[h] = acc;
                                              }
                                            });

                                            std::copy(next_vec.begin(), next_vec.end(), out_t);
                                            state_vec.swap(next_vec);
                                          }
                                        }
                                      });

  return outputs;
}

}  // namespace

#ifdef WITH_CUDA
at::Tensor linear_rnn_scan_cuda(const at::Tensor& weight_hh,
                                const at::Tensor& weight_xh,
                                const at::Tensor& bias,
                                const at::Tensor& inputs,
                                const at::Tensor& initial_state);
#endif

at::Tensor linear_rnn_scan(const at::Tensor& weight_hh,
                           const at::Tensor& weight_xh,
                           const at::Tensor& bias,
                           const at::Tensor& inputs,
                           const at::Tensor& initial_state) {
  if (inputs.size(0) == 0) {
    return at::empty({0, inputs.size(1), weight_hh.size(0)}, inputs.options());
  }

  if (inputs.is_cuda()) {
#ifdef WITH_CUDA
    return linear_rnn_scan_cuda(weight_hh, weight_xh, bias, inputs, initial_state);
#else
    TORCH_CHECK(false, "linear_rnn_scan CUDA extension was not built");
#endif
  }

  return linear_rnn_scan_cpu(weight_hh, weight_xh, bias, inputs, initial_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_rnn_scan", &linear_rnn_scan, "Linear RNN scan kernel");
}

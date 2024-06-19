#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor add_cuda_forward(torch::Tensor x, torch::Tensor y);
std::vector<torch::Tensor> add_cuda_backward(torch::Tensor grad_out);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CONTIGUOUS(x)

torch::Tensor add_forward(torch::Tensor x, torch::Tensor y) {
  // CHECK_INPUT(x);
  // CHECK_INPUT(y);
  return add_cuda_forward(x, y);
}

std::vector<torch::Tensor> add_backward(torch::Tensor grad_out) {
  // CHECK_INPUT(grad_out);
  return add_cuda_backward(grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward, "add forward (CUDA)");
  m.def("backward", &add_backward, "add backward (CUDA)");
}
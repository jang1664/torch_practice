#include <iostream>
#include <torch/extension.h>
#include <vector>

torch::Tensor myadd(torch::Tensor x, torch::Tensor y) { return x + y; }

torch::Tensor add_forward(torch::Tensor x, torch::Tensor y) { return myadd(x, y); }
std::vector<torch::Tensor> add_backward(torch::Tensor grad_out) { return {grad_out, grad_out}; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward, "add forward desc");
  m.def("backward", &add_backward, "add backward desc");
}
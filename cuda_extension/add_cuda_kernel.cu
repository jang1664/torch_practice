#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__global__ void add_cuda_forward_kernel(const scalar_t *__restrict__ x,
                                        const scalar_t *__restrict__ y, scalar_t *__restrict__ out,
                                        size_t N) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    out[index] = x[index] + y[index];
  }
}

template <typename scalar_t>
__global__ void add_cuda_backward_kernel(const scalar_t *__restrict__ out_grad,
                                         scalar_t *__restrict__ x_grad,
                                         scalar_t *__restrict__ y_grad, const int N) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    x_grad[index] = out_grad[index];
    y_grad[index] = out_grad[index];
  }
}

torch::Tensor add_cuda_forward(torch::Tensor x, torch::Tensor y) {

  const int n = x.size(0);
  const int threads = 1024;
  const dim3 blocks((n + threads - 1) / threads);

  torch::Tensor out = torch::zeros_like(x);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "add_forward_cuda", ([&] {
                               add_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                   x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(), n);
                             }));

  return out;
}

std::vector<torch::Tensor> add_cuda_backward(torch::Tensor out_grad) {
  const int n = out_grad.size(0);
  const int threads = 1024;
  const dim3 blocks((n + threads - 1) / threads);

  torch::Tensor x_grad = torch::zeros_like(out_grad);
  torch::Tensor y_grad = torch::zeros_like(out_grad);

  AT_DISPATCH_FLOATING_TYPES(out_grad.type(), "add_backward_cuda", ([&] {
                               add_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                   out_grad.data<scalar_t>(), x_grad.data<scalar_t>(),
                                   y_grad.data<scalar_t>(), n);
                             }));
  return {x_grad, y_grad};
}
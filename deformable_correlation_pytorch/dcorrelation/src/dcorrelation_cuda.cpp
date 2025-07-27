#include <torch/extension.h>
#include "dcorrelation_kernel.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor dcorrelation_forward(
    const at::Tensor& feat1,
    const at::Tensor& feat2,
    const at::Tensor& offset) {
    
    CHECK_INPUT(feat1);
    CHECK_INPUT(feat2);
    CHECK_INPUT(offset);

    TORCH_CHECK(feat1.sizes() == feat2.sizes(), "feat1 and feat2 must have the same size");
    TORCH_CHECK(offset.size(1) == 2, "offset must have 2 channels in the second dimension");
    TORCH_CHECK(feat1.size(0) == offset.size(0), "batch size must match");

    return dcorrelation_forward_cuda(feat1, feat2, offset);
}

std::vector<at::Tensor> dcorrelation_backward(
    const at::Tensor& grad_output,
    const at::Tensor& feat1,
    const at::Tensor& feat2,
    const at::Tensor& offset) {
    
    CHECK_INPUT(grad_output);
    CHECK_INPUT(feat1);
    CHECK_INPUT(feat2);
    CHECK_INPUT(offset);

    return dcorrelation_backward_cuda(grad_output, feat1, feat2, offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dcorrelation_forward, "Deformable Correlation Forward (CUDA)");
    m.def("backward", &dcorrelation_backward, "Deformable Correlation Backward (CUDA)");
}

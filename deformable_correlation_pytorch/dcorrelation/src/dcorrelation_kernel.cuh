#pragma once

#include <torch/extension.h>

// CUDA definitions for forward propagation
at::Tensor dcorrelation_forward_cuda(
    const at::Tensor& feat1,
    const at::Tensor& feat2,
    const at::Tensor& offset);

// CUDA definitions for backward propagation
std::vector<at::Tensor> dcorrelation_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& feat1,
    const at::Tensor& feat2,
    const at::Tensor& offset);

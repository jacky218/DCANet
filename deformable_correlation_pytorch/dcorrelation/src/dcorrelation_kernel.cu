#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h> 
#include "dcorrelation_kernel.cuh"
// CUDA device aassert, for debugging
#if defined(__CUDA_ARCH__)
#if defined(NDEBUG)
#define CUDA_KERNEL_ASSERT(x)
#else
#define CUDA_KERNEL_ASSERT(x) assert(x)
#endif
#else
#define CUDA_KERNEL_ASSERT(x)
#endif


// 双线性插值帮助函数
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_value_from_feat(
    const scalar_t* feat,
    const int height,
    const int width,
    const float y,
    const float x) {
    
    // 检查边界
    if (y <= -1 || y >= height || x <= -1 || x >= width) {
        return 0;
    }

    // 限制坐标在有效范围内
    const float y_real = fmaxf(0.0f, fminf((float)height - 1.0001f, y));
    const float x_real = fmaxf(0.0f, fminf((float)width - 1.0001f, x));

    const int y_low = floorf(y_real);
    const int x_low = floorf(x_real);
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;

    const float ly = y_real - y_low;
    const float lx = x_real - x_low;
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;

    const scalar_t v1 = feat[y_low * width + x_low];
    const scalar_t v2 = feat[y_low * width + x_high];
    const scalar_t v3 = feat[y_high * width + x_low];
    const scalar_t v4 = feat[y_high * width + x_high];

    const scalar_t w1 = hy * hx;
    const scalar_t w2 = hy * lx;
    const scalar_t w3 = ly * hx;
    const scalar_t w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}


// 前向传播 CUDA 核函数
template <typename scalar_t>
__global__ void dcorrelation_forward_kernel(
    const scalar_t* __restrict__ feat1,
    const scalar_t* __restrict__ feat2,
    const scalar_t* __restrict__ offset,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = batch_size * height * width;

    if (index >= total_pixels) return;

    const int w = index % width;
    const int h = (index / width) % height;
    const int b = index / (width * height);

    const int offset_b = b * height * width;
    const int offset_c = channels * height * width;

    // 获取偏移量
    const float offset_x = (float)offset[b * 2 * height * width + 0 * height * width + h * width + w];
    const float offset_y = (float)offset[b * 2 * height * width + 1 * height * width + h * width + w];

    // 计算 feat2 中的采样坐标
    const float sample_x = (float)w + offset_x;
    const float sample_y = (float)h + offset_y;

    scalar_t correlation_val = 0.0;

    // 对每个通道进行计算
    for (int c = 0; c < channels; ++c) {
        const scalar_t val1 = feat1[b * offset_c + c * height * width + h * width + w];
        const scalar_t val2 = get_value_from_feat(
            feat2 + b * offset_c + c * height * width, height, width, sample_y, sample_x);
        correlation_val += val1 * val2;
    }
    
    output[offset_b + h * width + w] = correlation_val;
}

// 后向传播帮助函数
template <typename scalar_t>
__device__ __forceinline__ void add_grad_to_feat(
    scalar_t* grad_feat,
    const int height,
    const int width,
    const float y,
    const float x,
    const scalar_t grad_val) {

    // 检查边界
    if (y <= -1 || y >= height || x <= -1 || x >= width) {
        return;
    }

    const float y_real = fmaxf(0.0f, fminf((float)height - 1.0001f, y));
    const float x_real = fmaxf(0.0f, fminf((float)width - 1.0001f, x));

    const int y_low = floorf(y_real);
    const int x_low = floorf(x_real);
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;

    const float ly = y_real - y_low;
    const float lx = x_real - x_low;
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;

    const scalar_t w1 = hy * hx;
    const scalar_t w2 = hy * lx;
    const scalar_t w3 = ly * hx;
    const scalar_t w4 = ly * lx;

    // 使用 atomicAdd 防止竞争条件
    atomicAdd(&grad_feat[y_low * width + x_low], w1 * grad_val);
    atomicAdd(&grad_feat[y_low * width + x_high], w2 * grad_val);
    atomicAdd(&grad_feat[y_high * width + x_low], w3 * grad_val);
    atomicAdd(&grad_feat[y_high * width + x_high], w4 * grad_val);
}

// 后向传播 CUDA 核函数
template <typename scalar_t>
__global__ void dcorrelation_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ feat1,
    const scalar_t* __restrict__ feat2,
    const scalar_t* __restrict__ offset,
    scalar_t* __restrict__ grad_feat1,
    scalar_t* __restrict__ grad_feat2,
    scalar_t* __restrict__ grad_offset,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = batch_size * height * width;

    if (index >= total_pixels) return;

    const int w = index % width;
    const int h = (index / width) % height;
    const int b = index / (width * height);

    const int offset_b = b * height * width;
    const int offset_c = channels * height * width;

    // 获取输出梯度
    const scalar_t grad_out_val = grad_output[offset_b + h * width + w];
    
    // 获取偏移量
    const float offset_x = (float)offset[b * 2 * height * width + 0 * height * width + h * width + w];
    const float offset_y = (float)offset[b * 2 * height * width + 1 * height * width + h * width + w];
    
    // 计算采样坐标
    const float sample_x = (float)w + offset_x;
    const float sample_y = (float)h + offset_y;

    scalar_t grad_offset_x = 0.0;
    scalar_t grad_offset_y = 0.0;

    for (int c = 0; c < channels; ++c) {
        const int channel_offset = b * offset_c + c * height * width;
        const scalar_t* feat2_c = feat2 + channel_offset;
        const scalar_t f1_val = feat1[channel_offset + h * width + w];
        
        // 计算 feat2 的插值
        const scalar_t f2_interp_val = get_value_from_feat(feat2_c, height, width, sample_y, sample_x);

        // 1. 计算 grad_feat1
        // dL/df1 = dL/d_out * d_out/df1 = grad_out * f2_interp
        grad_feat1[channel_offset + h * width + w] = grad_out_val * f2_interp_val;
        
        // 2. 计算 grad_feat2
        // dL/df2 = dL/d_out * d_out/df2 = grad_out * f1
        // 这个梯度需要被分配到用于插值的4个点上
        add_grad_to_feat(grad_feat2 + channel_offset, height, width, sample_y, sample_x, grad_out_val * f1_val);
        
        // 3. 计算 grad_offset
        // dL/d_offset = dL/d_out * d_out/d_offset = grad_out * f1 * df2_interp/d_offset
        // df2_interp/d_offset_x = df2_interp/dsample_x
        // df2_interp/d_offset_y = df2_interp/dsample_y
        // 使用中心差分计算 feat2 的空间梯度
        const scalar_t gx = 0.5 * (get_value_from_feat(feat2_c, height, width, sample_y, sample_x + 1) -
                                   get_value_from_feat(feat2_c, height, width, sample_y, sample_x - 1));
        const scalar_t gy = 0.5 * (get_value_from_feat(feat2_c, height, width, sample_y + 1, sample_x) -
                                   get_value_from_feat(feat2_c, height, width, sample_y - 1, sample_x));

        grad_offset_x += grad_out_val * f1_val * gx;
        grad_offset_y += grad_out_val * f1_val * gy;
    }
    
    grad_offset[b * 2 * height * width + 0 * height * width + h * width + w] = grad_offset_x;
    grad_offset[b * 2 * height * width + 1 * height * width + h * width + w] = grad_offset_y;
}


// C++ -> CUDA 启动函数
at::Tensor dcorrelation_forward_cuda(
    const at::Tensor& feat1,
    const at::Tensor& feat2,
    const at::Tensor& offset) {
    
    const auto batch_size = feat1.size(0);
    const auto channels = feat1.size(1);
    const auto height = feat1.size(2);
    const auto width = feat1.size(3);

    auto output = at::zeros({batch_size, height, width}, feat1.options());
    
    const int threads = 1024;
    const int blocks = (batch_size * height * width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(feat1.scalar_type(), "dcorrelation_forward_cuda", ([&] {
        dcorrelation_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            feat1.data_ptr<scalar_t>(),
            feat2.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels, height, width
        );
    }));
    
    return output;
}


std::vector<at::Tensor> dcorrelation_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& feat1,
    const at::Tensor& feat2,
    const at::Tensor& offset) {
        
    const auto batch_size = feat1.size(0);
    const auto channels = feat1.size(1);
    const auto height = feat1.size(2);
    const auto width = feat1.size(3);

    auto grad_feat1 = at::zeros_like(feat1);
    auto grad_feat2 = at::zeros_like(feat2);
    auto grad_offset = at::zeros_like(offset);
    
    const int threads = 1024;
    const int blocks = (batch_size * height * width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(feat1.scalar_type(), "dcorrelation_backward_cuda", ([&] {
        dcorrelation_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.data_ptr<scalar_t>(),
            feat1.data_ptr<scalar_t>(),
            feat2.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            grad_feat1.data_ptr<scalar_t>(),
            grad_feat2.data_ptr<scalar_t>(),
            grad_offset.data_ptr<scalar_t>(),
            batch_size, channels, height, width
        );
    }));

    return {grad_feat1, grad_feat2, grad_offset};
}

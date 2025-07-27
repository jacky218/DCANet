# 导入 C++ 扩展
import torch
from torch.utils.cpp_extension import load

# JIT (Just-In-Time) compilation
_C = load(
    name='dcorrelation_cpp',
    sources=[
        '/home/honsen/conda_envs/TFNet-main/deformable_correlation_pytorch/dcorrelation/src/dcorrelation_cuda.cpp',
        '/home/honsen/conda_envs/TFNet-main/deformable_correlation_pytorch/dcorrelation/src/dcorrelation_kernel.cu',
    ],
    verbose=True
)

# 导入 Python 模块
from .modules.dcorrelation import DeformableCorrelation, MultiScaleDeformableCorrelation

__all__ = ['DeformableCorrelation', 'MultiScaleDeformableCorrelation']

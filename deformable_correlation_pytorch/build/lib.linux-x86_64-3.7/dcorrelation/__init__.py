# 导入 C++ 扩展
import torch
from torch.utils.cpp_extension import load

# JIT (Just-In-Time) compilation
_C = load(
    name='dcorrelation_cpp',
    sources=[
        'dcorrelation/src/dcorrelation_cuda.cpp',
        'dcorrelation/src/dcorrelation_kernel.cu',
    ],
    verbose=True
)

# 导入 Python 模块
from .modules.dcorrelation import DeformableCorrelation, MultiScaleDeformableCorrelation

__all__ = ['DeformableCorrelation', 'MultiScaleDeformableCorrelation']

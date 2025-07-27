from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dcorrelation',
    ext_modules=[
        CUDAExtension('dcorrelation._C', [
            '/home/honsen/conda_envs/TFNet-main/deformable_correlation_pytorch/dcorrelation/src/dcorrelation_cuda.cpp',
            '/home/honsen/conda_envs/TFNet-main/deformable_correlation_pytorch/dcorrelation/src/dcorrelation_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['dcorrelation', 'dcorrelation.functions', 'dcorrelation.modules'],
)

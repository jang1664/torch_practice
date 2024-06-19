from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='myadd',
    ext_modules=[
        CUDAExtension('myadd', [
            'add_cuda.cpp',
            'add_cuda_kernel_v2.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
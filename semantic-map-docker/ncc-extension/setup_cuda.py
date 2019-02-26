
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ncc',
    ext_modules=[
        CUDAExtension('ncc_cuda', [
            'ncc_cuda.cpp',
            'ncc_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })



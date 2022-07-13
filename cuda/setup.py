from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cuda',
    ext_modules=[
        CUDAExtension(name='lltm_cuda', # extension name, import this to use CUDA API
                      sources=[
                          'lltm_cuda.cpp',
                          'lltm_cuda_kernel.cu',
                      ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='lltm_cpp', # package name, import this to use python API
    ext_modules=[
        CppExtension(name='lltm_cpp', # extension name, import this to use cpp API
        sources=['lltm.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

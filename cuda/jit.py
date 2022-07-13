import os
from torch.utils.cpp_extension import load
_src_path = os.path.dirname(os.path.abspath(__file__))
lltm_cuda = load(
    'lltm_cuda', [os.path.join(_src_path, f) for f in ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu']], verbose=True)
# help(lltm_cuda)

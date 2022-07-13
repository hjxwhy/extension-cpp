import os
from torch.utils.cpp_extension import load
_src_path = os.path.dirname(os.path.abspath(__file__)) ## if do not set the absolute path, it can't find the source file
lltm_cpp = load(name="lltm_cpp", sources=[os.path.join(_src_path, f) for f in ["lltm.cpp"]], verbose=True)
# help(lltm_cpp)

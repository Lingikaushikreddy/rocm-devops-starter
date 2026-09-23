from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_ops",
    ext_modules=[CUDAExtension("fused_ops", ["fused_ops.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)

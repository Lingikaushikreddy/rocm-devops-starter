from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fused_ops",
    ext_modules=[CppExtension("fused_ops", ["fused_ops.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)

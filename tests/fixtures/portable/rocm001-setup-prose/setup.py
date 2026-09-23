from setuptools import setup

print("For fused kernels, pip install apex from https://github.com/NVIDIA/apex")
setup(name="demo", extras_require={"apex": ["torch"]})

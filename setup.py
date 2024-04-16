# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
from pathlib import Path

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME




ext_modules = [] 
ext_modules.append(
    CUDAExtension(
        name="mini-flashattn",
        sources=[
            "hello.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] #+ generator_flag,
            "nvcc": 
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo"
                ]
                # + generator_flag
                # + cc_flag
            ,
        },
        include_dirs=[
        ],
    )
)

setup(
    name="mini-flashatten",
    version="0.0.1",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info",)
    ),
    author="Xws",
    author_email="shamy117@qq.com",
    description="Minimize implementation of Flash Attention",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xws117/mini-flashattention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
    ],
)

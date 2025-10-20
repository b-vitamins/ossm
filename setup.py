from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension, CUDAExtension,
                                       CUDA_HOME)

root = Path(__file__).resolve().parent
src_dir = root / "src" / "ossm" / "csrc"

cpp_sources = sorted(str(path) for path in src_dir.glob("*.cpp"))
cuda_sources = sorted(str(path) for path in src_dir.glob("*.cu"))

use_cuda = bool(cuda_sources) and CUDA_HOME is not None
sources = cpp_sources + (cuda_sources if use_cuda else [])
extension_cls = CUDAExtension if use_cuda else CppExtension

extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

if use_cuda:
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-Xcudafe",
        "--diag_suppress=20012",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-allow-unsupported-compiler",
    ]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list:
        token = arch_list.split()[0].replace(".", "")
        if token.isdigit():
            nvcc_flags.append(f"-gencode=arch=compute_{token},code=sm_{token}")
    extra_compile_args["nvcc"] = nvcc_flags

extension_kwargs = dict(
    name="ossm._kernels",
    sources=sources,
    include_dirs=[str(src_dir)],
    extra_compile_args=extra_compile_args,
)

if use_cuda:
    extension_kwargs.setdefault("define_macros", []).append(("WITH_CUDA", None))

setup(
    name="ossm_kernels",
    package_dir={'': 'src'},
    ext_modules=[extension_cls(**extension_kwargs)],
    cmdclass={"build_ext": BuildExtension},
)

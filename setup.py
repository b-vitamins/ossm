from __future__ import annotations

import os
import re
from pathlib import Path

import tomllib

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

root = Path(__file__).resolve().parent
src_dir = root / "src" / "ossm" / "csrc"


def _gather_sources(*patterns: str) -> list[str]:
    files: set[str] = set()
    for pattern in patterns:
        files.update(_relative_posix(path) for path in src_dir.glob(pattern))
    return sorted(files)

def _relative_posix(path: Path) -> str:
    """Return a path relative to the project root using POSIX separators."""

    return path.relative_to(root).as_posix()


def read_version() -> str:
    pyproject = root / "pyproject.toml"

    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["version"]


version = read_version()
suffix = os.environ.get("OSSM_BUILD_SUFFIX", "").strip()
if suffix:
    normalized_suffix = re.sub(r"[^0-9A-Za-z.-]", "-", suffix)
    version = f"{version}+{normalized_suffix}"

cpp_sources = _gather_sources("*.cpp", "sdlinoss_fast/**/*.cpp")
cuda_sources = _gather_sources("*.cu", "sdlinoss_fast/**/*.cu")

use_cuda = bool(cuda_sources) and CUDA_HOME is not None
sources = cpp_sources + (cuda_sources if use_cuda else [])
extension_cls = CUDAExtension if use_cuda else CppExtension

extra_compile_args = {"cxx": ["-O3", "-std=c++17", "-march=native", "-DOSSM_FAST=1"]}
extra_link_args: list[str] = []

if os.name == "nt":
    extra_compile_args["cxx"].extend(["/fp:fast", "/openmp"])
else:
    extra_compile_args["cxx"].extend(
        ["-ffast-math", "-fno-math-errno", "-fno-trapping-math", "-fopenmp"]
    )
    extra_link_args.append("-fopenmp")

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
    nvcc_flags.append("-DOSSM_FAST=1")
    if os.environ.get("OSSM_SDLINOSS_FAST_PREFETCH") == "1":
        nvcc_flags.append("-DOSSM_FAST_PREFETCH=1")
    nvcc_flags.append(
        f"-DOSSM_FAST_UNROLL={int(os.getenv('OSSM_FAST_UNROLL', '2'))}"
    )
    extra_compile_args["nvcc"] = nvcc_flags

extension_kwargs = dict(
    name="ossm._kernels",
    sources=sources,
    include_dirs=[_relative_posix(src_dir)],
    extra_compile_args=extra_compile_args,
)

if extra_link_args:
    extension_kwargs["extra_link_args"] = extra_link_args

if use_cuda:
    extension_kwargs.setdefault("define_macros", []).append(("WITH_CUDA", None))

setup(
    name="ossm_kernels",
    version=version,
    package_dir={'': 'src'},
    ext_modules=[extension_cls(**extension_kwargs)],
    cmdclass={"build_ext": BuildExtension},
    setup_requires=["wheel"],
)


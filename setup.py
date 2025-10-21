from __future__ import annotations

import os
import re
from pathlib import Path

import subprocess
import sys

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


def _ensure_wheel() -> None:
    """Ensure the wheel package is available when setuptools needs it."""

    try:
        import wheel  # noqa: F401  # pragma: no cover
    except ModuleNotFoundError:  # pragma: no cover - best-effort safeguard
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wheel"])


_ensure_wheel()


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

cpp_sources = sorted(_relative_posix(path) for path in src_dir.glob("*.cpp"))
cuda_sources = sorted(_relative_posix(path) for path in src_dir.glob("*.cu"))

use_cuda = bool(cuda_sources) and CUDA_HOME is not None
sources = cpp_sources + (cuda_sources if use_cuda else [])
extension_cls = CUDAExtension if use_cuda else CppExtension

extra_compile_args = {"cxx": ["-O3", "-std=c++17", "-march=native"]}

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
    include_dirs=[_relative_posix(src_dir)],
    extra_compile_args=extra_compile_args,
)

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


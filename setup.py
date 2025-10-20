from __future__ import annotations

import os
import re
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension, CUDAExtension,
                                       CUDA_HOME)

root = Path(__file__).resolve().parent
src_dir = root / "src" / "ossm" / "csrc"


def read_version() -> str:
    pyproject = root / "pyproject.toml"

    try:  # Python ≥3.11
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.9/3.10
        try:
            import tomli as tomllib  # type: ignore[assignment]
        except ModuleNotFoundError as exc:  # pragma: no cover - final fallback
            text = pyproject.read_text(encoding="utf-8")
            match = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
            if match is None:
                raise RuntimeError("Unable to determine version from pyproject.toml") from exc
            return match.group(1)
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["version"]


version = read_version()
suffix = os.environ.get("OSSM_BUILD_SUFFIX", "").strip()
if suffix:
    normalized_suffix = re.sub(r"[^0-9A-Za-z.-]", "-", suffix)
    version = f"{version}+{normalized_suffix}"

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
    version=version,
    package_dir={'': 'src'},
    ext_modules=[extension_cls(**extension_kwargs)],
    cmdclass={"build_ext": BuildExtension},
)

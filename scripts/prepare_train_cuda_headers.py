#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Create an opt-in CUDA 12.8 header overlay for recent glibc; never edit CUDA itself."""
import argparse
import os
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.cuda_root / "include").resolve()
    output = args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error("output must be separate from the CUDA installation")
    version = re.search(r"^#define CUDA_VERSION\s+(\d+)", (source / "cuda.h").read_text(), re.MULTILINE)
    if not version or version[1] != "12080":
        parser.error("this compatibility overlay is only for CUDA 12.8")
    glibc = tuple(map(int, os.confstr("CS_GNU_LIBC_VERSION").split()[1].split(".")))
    names = []
    if glibc >= (2, 41):
        names += ["sinpi", "sinpif", "cospi", "cospif"]
    if glibc >= (2, 43):
        names += ["rsqrt", "rsqrtf"]
    text = (source / "crt/math_functions.h").read_text()
    for name in names:
        text, count = re.subn(
            rf"^(extern __DEVICE_FUNCTIONS_DECL__[^\n]*\b{name}\([^\n;]*\));$",
            r"\1 noexcept(true);", text, flags=re.MULTILINE)
        if count != 1:
            parser.error(f"expected one unpatched CUDA declaration for {name}, found {count}")
    shutil.copytree(source, output, dirs_exist_ok=True)
    (output / "crt/math_functions.h").write_text(text)
    print(f"Prepared {output}; adjusted {len(names)} declarations for glibc {'.'.join(map(str, glibc))}")
    print(f"Configure with -DCMAKE_CUDA_FLAGS=-I{output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject desktop sources/packages/runtime libraries in a configured train build."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

FORBIDDEN_SOURCE = re.compile(
    r"/src/(app|visualizer|rendering|sequencer|python|mcp|tcp|preprocessing)/"
    r"|/src/core/nn/models/(?!lpips\.cpp$)"
    r"|/src/training/normal_auto_generate\.cpp"
    r"|/src/io/(video/|video_|hdr_|mesh/|formats/(sogs|spz|rad|usd|html|nurec_usdz))"
    r"|/external/(zep|tree-sitter|tree-sitter-python|spz|tinyusdz)/")
FORBIDDEN_PACKAGE = re.compile(r"^(sdl3|vulkan.*|volk|rmlui|freetype|ffmpeg|libplacebo|python3|nanobind|cppzmq|zeromq|assimp|shader-slang|glslang)$")
FORBIDDEN_LIBRARY = re.compile(r"SDL|vulkan|RmlUi|freetype|libavcodec|libavformat|libavfilter|placebo|libpython", re.I)


def audit(build: Path, executable: Path | None):
    failures = []
    commands = json.loads((build / "compile_commands.json").read_text())
    if not commands:
        failures.append("compile_commands.json is empty")
    for command in commands:
        source = Path(command["file"])
        if not source.is_absolute():
            source = Path(command["directory"]) / source
        if FORBIDDEN_SOURCE.search(source.as_posix()):
            failures.append(f"Forbidden compilation unit: {source}")
    status = build / "vcpkg_installed/vcpkg/status"
    if status.exists():
        for package in re.findall(r"^Package: (.+)$", status.read_text(), re.M):
            if FORBIDDEN_PACKAGE.fullmatch(package):
                failures.append(f"Forbidden installed dependency: {package}")
    else:
        failures.append(f"Missing manifest dependency inventory: {status}")
    if executable:
        if sys.platform.startswith("linux"):
            result = subprocess.run(["ldd", str(executable.resolve())], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append(f"ldd failed: {result.stdout.strip()}")
            for line in result.stdout.splitlines():
                if "not found" in line or FORBIDDEN_LIBRARY.search(line):
                    failures.append(f"Invalid runtime dependency: {line.strip()}")
        else:
            failures.append("Runtime dependency audit currently requires Linux ldd")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    try:
        failures = audit(args.build, args.executable)
    except (OSError, ValueError) as error:
        failures = [str(error)]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Train build audit passed: no desktop compilation units or installed packages"
          + ("; runtime dependencies resolved" if args.executable else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

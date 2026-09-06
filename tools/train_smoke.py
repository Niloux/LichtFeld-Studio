#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small GPU regression for lfs-train: COLMAP, PLY, project resume and capability errors.

Uses only the Python standard library unless --jpeg is requested (requires Pillow).
All generated inputs, outputs and logs live under a new --output directory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib


def write_png(path, width, height, channels, pixels):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + bytes(pixels[y * width * channels:(y + 1) * width * channels]) for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def make_scene(root, jpeg=False):
    images = root / "images"
    sparse = root / "sparse/0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    width = height = 64
    (sparse / "cameras.txt").write_text("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n1 PINHOLE 64 64 60 60 32 32\n")
    points = [(i + 1, (i % 16 - 7.5) * 0.055, (i // 16 - 7.5) * 0.055, 2.0 + 0.04 * math.sin(i)) for i in range(256)]
    centers = [(-0.15, -0.05), (0.15, -0.05), (-0.10, 0.08), (0.10, 0.08)]
    camera_lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    for camera, (cx, cy) in enumerate(centers, 1):
        name = f"image_{camera:03d}.{'jpg' if jpeg else 'png'}"
        pixels = [value for y in range(height) for x in range(width)
                  for value in (40 + x * 2, 40 + y * 2, 70 + (x + y) // 2)]
        if jpeg:
            from PIL import Image
            Image.frombytes("RGB", (width, height), bytes(pixels)).save(images / name, quality=95)
        else:
            write_png(images / name, width, height, 3, pixels)
        camera_lines.append(f"{camera} 1 0 0 0 {-cx} {-cy} 0 1 {name}")
        camera_lines.append(" ".join(f"{60*(x-cx)/z+32} {60*(y-cy)/z+32} {pid}" for pid, x, y, z in points))
    (sparse / "images.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK\n" + "\n".join(
        f"{pid} {x} {y} {z} 128 128 128 0.1 " + " ".join(f"{c} {pid-1}" for c in range(1, 5))
        for pid, x, y, z in points) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jpeg", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "dataset"
    make_scene(dataset, args.jpeg)
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    runs = []

    def run(name, options, error=None):
        start = time.monotonic()
        result = subprocess.run([str(executable), *map(str, options)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        log = result.stdout.decode("utf-8", errors="replace")
        (root / f"{name}.log").write_text(log)
        entry = {"name": name, "returncode": result.returncode, "seconds": round(time.monotonic()-start, 3)}
        runs.append(entry)
        (root / "report.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
        if (error is None and (result.returncode != 0 or "[error]" in log or "LFS FAILURE REPORT" in log)) or (error is not None and (result.returncode == 0 or error not in log)):
            raise RuntimeError(f"{name} failed; see {root / (name + '.log')}\n{log[-3000:]}")
        return log

    base = ["-d", dataset, "--max-cap", "1024", "--sh-degree", "0"]
    run("fresh", [*base, "-o", root / "fresh", "--iter", "20", "--export", "ply"])
    project = root / "fresh/project.licht"
    assert project.is_file() and project.stat().st_size > 0, "Missing durable training project"
    assert (root / "fresh/splat_20.ply").is_file(), "Missing final PLY at iteration 20"
    run("resume", ["--resume", project, "-o", root / "resume", "--iter", "24", "--export", "ply"])
    assert (root / "resume/project.licht").is_file(), "Missing resumed project"
    assert (root / "resume/splat_24.ply").is_file(), "Resume did not reach iteration 24"
    # Exercise refinement of quantized SH-rest buffers and their process-exit
    # cleanup. A successful exit code alone does not catch teardown diagnostics.
    run("q16_refine_exit", ["-d", dataset, "--max-cap", "1024", "--sh-degree", "3",
                            "--sh-degree-interval", "100", "--iter", "600",
                            "-o", root / "q16", "--export", "ply"])
    assert (root / "q16/splat_600.ply").is_file(), "Missing refined Q16 export"
    run("unsupported_export", [*base, "-o", root / "bad-export", "--export", "spz"], "only --export ply")
    run("unsupported_script", [*base, "-o", root / "bad-script", "--python-script", "absent.py"], "Python scripts are unavailable")
    run("unsupported_video", ["--render-output", "absent.mp4"], "video options are unavailable")
    for kind in ("depth", "normal"):
        run(f"missing_{kind}", [*base, "-o", root / f"missing-{kind}", "--iter", "1", f"--use-{kind}-loss"], f"Required {kind} map missing")
    run("missing_mask", [*base, "-o", root / "missing-mask", "--iter", "1", "--mask-mode", "segment"], "Required mask or image alpha missing")
    print(f"Smoke tests passed; report: {root / 'report.json'}")


if __name__ == "__main__":
    main()

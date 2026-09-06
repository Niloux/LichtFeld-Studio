#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small GPU smoke test for sky training, resume, foreground PLY and validation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from train_smoke import make_scene, write_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "dataset"
    make_scene(dataset)
    (dataset / "sky_masks").mkdir()
    (dataset / "masks").mkdir()
    for path in (dataset / "images").iterdir():
        write_png(dataset / "sky_masks" / (path.name + ".png"), 64, 64, 1,
                  [255 if y < 22 else 0 for y in range(64) for x in range(64)])
        write_png(dataset / "masks" / path.name, 64, 64, 1,
                  [255 if x > 2 else 0 for y in range(64) for x in range(64)])
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    reports = []

    def run(name, options, error=None):
        result = subprocess.run([str(args.executable.resolve()), *map(str, options)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        log = result.stdout.decode(errors="replace")
        (root / (name + ".log")).write_text(log)
        reports.append({"name": name, "exit": result.returncode})
        (root / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
        if error:
            assert result.returncode != 0 and error in log, log[-4000:]
        else:
            assert result.returncode == 0 and "[error]" not in log and "LFS FAILURE REPORT" not in log, log[-4000:]
        return log

    base = ["-d", dataset, "--max-cap", 1024, "--sh-degree", 0, "--sky", "--sky-num-points", 1024,
            "--mask-mode", "ignore", "--export", "ply"]
    for backend, flags in (("gut", ["--gut", "--exposure-correction", "--eval", "--eval-steps", 10, "--test-every", 2]), ("fast", [])):
        run(backend, [*base, *flags, "-o", root / backend, "--iter", 20])
        if backend == "gut":
            assert len((root / backend / "metrics.csv").read_text().splitlines()) > 1, "Evaluation did not run"
        project = root / backend / "project.licht"
        assert project.is_file()
        ply = (root / backend / "splat_20.ply").read_bytes().split(b"end_header", 1)[0]
        assert re.search(rb"element vertex 256\s", ply), "Sky must not add Gaussian rows"
        log = run(backend + "_resume", ["--resume", project, "-o", root / (backend + "_resume"),
                                        "--iter", 24, "--export", "ply"])
        assert "optimizer step 20" in log, "Sky optimizer state was not restored"
        assert (root / (backend + "_resume") / "splat_24.ply").is_file()
    run("wrong_resume_geometry", ["--resume", root / "gut/project.licht", "-o", root / "wrong-resume",
                                    "--iter", 24, "--sky-num-points", 2048], "Cannot change Gaussian sky geometry")
    run("missing_sky_mask", [*base, "-o", root / "missing", "--iter", 1,
                             "--sky-mask-dir", "missing_sky"], "Missing sky mask")
    run("invalid_opacity", [*base, "-o", root / "bad-boundary", "--iter", 1,
                             "--sky-initial-opacity", 1.5], "Invalid Gaussian sky geometry")
    # Feature-off still loads and saves the old checkpoint layout.
    run("off", ["-d", dataset, "-o", root / "off", "--max-cap", 1024, "--sh-degree", 0, "--iter", 2])
    run("off_resume", ["--resume", root / "off/project.licht", "-o", root / "off-resume", "--iter", 3])
    print(f"Sky smoke tests passed: {root / 'report.json'}")


if __name__ == "__main__":
    main()

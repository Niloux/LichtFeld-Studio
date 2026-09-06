#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise profile selection before project(), without CUDA or package installation."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrainProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lfs-profile-")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.build = Path(self.temp.name) / "build"
        self.source.mkdir()
        (self.source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.30)
include("{ROOT.as_posix()}/cmake/BuildProfile.cmake")
# Capture exactly what the toolchain would see at the first project().
file(WRITE "${{CMAKE_BINARY_DIR}}/selection.txt"
    "${{LFS_TRAIN_ONLY}}|${{VCPKG_MANIFEST_FEATURES}}|${{VCPKG_MANIFEST_NO_DEFAULT_FEATURES}}|${{LFS_ENABLE_VULKAN_VALIDATION}}")
project(ProfileProbe LANGUAGES NONE)
''')

    def configure(self, *options):
        return subprocess.run(["cmake", "-S", str(self.source), "-B", str(self.build), *options],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_default_preserves_studio_dependencies(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout)
        values = (self.build / "selection.txt").read_text().split("|")
        self.assertEqual(values[0], "OFF")
        self.assertIn("studio", values[1].split(";"))

    def test_train_excludes_desktop_features_even_in_debug(self):
        result = self.configure("-DLFS_BUILD_PROFILE=train", "-DCMAKE_BUILD_TYPE=Debug",
                                "-DVCPKG_MANIFEST_FEATURES=studio;vulkan-validation;tests",
                                "-DLFS_ENABLE_VULKAN_VALIDATION=ON")
        self.assertEqual(result.returncode, 0, result.stdout)
        values = (self.build / "selection.txt").read_text().split("|")
        self.assertEqual(values, ["ON", "tests", "ON", "OFF"])

    def test_unknown_profile_rejected(self):
        result = self.configure("-DLFS_BUILD_PROFILE=typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be studio or train", result.stdout)

    def test_unimplemented_integrations_rejected(self):
        for option in ("LFS_TRAIN_PYTHON_LOSS", "LFS_TRAIN_PREPROCESS"):
            result = self.configure("-DLFS_BUILD_PROFILE=train", f"-D{option}=ON")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not implemented", result.stdout)
            self.configure("-DLFS_BUILD_PROFILE=train", f"-D{option}=OFF")

    def test_profile_change_requires_new_directory(self):
        self.assertEqual(self.configure().returncode, 0)
        result = self.configure("-DLFS_BUILD_PROFILE=train")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new build directory", result.stdout)

    def test_manifest_has_no_desktop_base_dependencies(self):
        manifest = json.loads((ROOT / "vcpkg.json").read_text())
        names = {d if isinstance(d, str) else d["name"] for d in manifest["dependencies"]}
        forbidden = {"sdl3", "vulkan", "ffmpeg", "libplacebo", "rmlui", "python3", "nanobind", "cppzmq", "assimp", "freetype"}
        self.assertFalse(names & forbidden)
        studio = manifest["features"]["studio"]["dependencies"]
        studio_names = {d if isinstance(d, str) else d["name"] for d in studio}
        self.assertTrue(forbidden <= studio_names)


if __name__ == "__main__":
    unittest.main()

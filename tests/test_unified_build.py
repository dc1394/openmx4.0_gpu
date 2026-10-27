#!/usr/bin/env python3
"""Verify CUDA/HIP source selection and build isolation without GPU toolchains.

Real repository inputs are projected into temporary directories. Native make
is exercised only with non-recursive object dry runs. Independent C adapters
are syntax checked; no dependency downloads or builds are made.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "tools" / "prepare_build.py"
HIP = ROOT / "source" / "backends" / "hip"
COMPILE_SUFFIXES = {".c", ".h", ".hpp", ".cu", ".cuh", ".cpp", ".f", ".f90", ".F90"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnifiedBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="openmx-unified-build-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.workspace = Path(cls.temporary.name)
        cls.manifest = json.loads((HIP / "sources.json").read_text())
        cls.builds = {}
        cls.stages = {}
        for backend in ("CUDA", "HIP"):
            build = cls.workspace / backend.lower()
            completed = cls.prepare(build, backend)
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            cls.builds[backend] = build
            cls.stages[backend] = build / "native" / "source"

    @staticmethod
    def prepare(build, backend):
        return subprocess.run(
            [sys.executable, str(PREPARE), "--source-root", str(ROOT),
             "--build-root", str(build), "--backend", backend],
            capture_output=True, text=True, check=False,
        )

    def test_import_manifest_preserves_all_amd_sources(self):
        self.assertEqual(self.manifest["backend"], "HIP")
        shared = set(self.manifest["shared"])
        overrides = self.manifest["overrides"]
        self.assertFalse(shared.intersection(overrides), "ambiguous HIP source selection")

        self.assertEqual(shared | set(overrides), set(self.manifest["import_sha256"]))
        for relative, expected in self.manifest["import_sha256"].items():
            with self.subTest(source=relative):
                source = HIP / overrides[relative] if relative in overrides else ROOT / "source" / relative
                self.assertTrue(source.is_file(), f"missing imported AMD source: {source}")
                self.assertEqual(digest(source), expected)
        for relative, imported in self.manifest["auxiliary"].items():
            with self.subTest(auxiliary=relative):
                original = HIP / imported
                projected = self.builds["HIP"] / "native" / relative
                self.assertTrue(original.is_file())
                self.assertEqual(digest(projected), digest(original))

    def test_cpu_projection_has_no_vendor_dependency_builds(self):
        build = self.workspace / 'cpu-only'
        result = self.prepare(build, 'CPU')
        self.assertEqual(result.returncode, 0, result.stderr)
        stage = build / 'native/source'
        self.assertEqual((stage / 'Set_Hamiltonian.c').resolve(), ROOT / 'source/Set_Hamiltonian.c')
        self.assertTrue((stage / 'cpu/include/openmx_cpu_runtime.h').is_file())
        self.assertFalse((stage / 'Makefile').exists())
        for name in ('GEMMul8', 'fftw3', 'magma', 'gpusolver2-install'):
            self.assertFalse((stage / 'third_party' / name).exists())
        rejected = self.prepare(build, 'CUDA')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('another backend', rejected.stderr)

    @unittest.skipUnless(shutil.which("cc"), "C compiler is required")
    def test_density_adapters_compile_in_all_projections(self):
        stages = dict(self.stages)
        build = self.workspace / 'density-cpu'
        result = self.prepare(build, 'CPU')
        self.assertEqual(result.returncode, 0, result.stderr)
        stages['CPU'] = build / 'native/source'
        for backend, stage in stages.items():
            with self.subTest(backend=backend):
                result = subprocess.run(
                    ['cc', '-std=c11', '-Werror', '-fsyntax-only',
                     str(stage / 'KumoF.c'), str(stage / 'Dr_KumoF.c')],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_selects_production_sources(self):
        cuda, hip = self.stages["CUDA"], self.stages["HIP"]
        integrated_overrides = self.manifest.get("integrated_overrides", {})
        self.assertFalse(set(integrated_overrides).intersection(self.manifest["overrides"]))
        self.assertFalse(set(integrated_overrides).intersection(self.manifest["shared"]))
        self.assertFalse(set(integrated_overrides).intersection(self.manifest["import_sha256"]))
        overrides = {**self.manifest["overrides"], **integrated_overrides}
        for name, imported in overrides.items():
            with self.subTest(override=name):
                self.assertEqual((hip / name).resolve(), (HIP / imported).resolve())
                if (ROOT / "source" / name).is_file() and Path(name).suffix in COMPILE_SUFFIXES:
                    self.assertEqual((cuda / name).resolve(), (ROOT / "source" / name).resolve())
                else:
                    self.assertFalse((cuda / name).exists(), f"HIP-only source leaked into CUDA: {name}")
        for name in self.manifest["shared"]:
            if Path(name).suffix in COMPILE_SUFFIXES:
                with self.subTest(shared=name):
                    self.assertEqual(digest(hip / name), self.manifest["import_sha256"][name])
        # Refactored implementations have one canonical copy, selected by both
        # backends; they are no longer immutable AMD import snapshots.
        integrated = self.manifest.get("integrated_shared", [])
        self.assertEqual(len(integrated), len(set(integrated)))
        self.assertFalse(set(integrated).intersection(self.manifest["shared"]))
        self.assertFalse(set(integrated).intersection(overrides))
        for name in integrated:
            with self.subTest(integrated=name):
                canonical = ROOT / "source" / name
                self.assertEqual((cuda / name).resolve(), canonical)
                self.assertEqual((hip / name).resolve(), canonical)
                self.assertFalse((HIP / "src" / name).exists())
                self.assertFalse((HIP / "include" / name).exists())
        amd_sources = (set(self.manifest["shared"]) | set(overrides) |
                       set(self.manifest.get("integrated_shared", [])))
        for source in (ROOT / "source").iterdir():
            if source.is_file() and source.suffix in COMPILE_SUFFIXES and source.name not in amd_sources:
                self.assertFalse((hip / source.name).exists(), f"CUDA-only source leaked into HIP: {source.name}")
        self.assertEqual((cuda / "Makefile").resolve(), ROOT / "source" / "Makefile")
        self.assertEqual((hip / "Makefile").resolve(), HIP / "Makefile")
        for stage in (cuda, hip):
            self.assertFalse((stage / "elpa-2018.05.001").exists())
            self.assertFalse((stage / "elpa1.f90").exists())
            self.assertTrue((stage / "openmx_elpa_cpu.c").is_file())

    def test_hip_compatibility_headers_do_not_shadow_cuda(self):
        headers = [name for name, source in self.manifest["overrides"].items()
                   if source.startswith("include/") and not (ROOT / "source" / name).exists()]
        self.assertTrue(headers, "HIP compatibility headers were not declared")
        for name in headers:
            with self.subTest(header=name):
                self.assertTrue((self.stages["HIP"] / name).is_file())
                self.assertFalse((self.stages["CUDA"] / name).exists())

    def test_mutable_dependency_trees_and_artifacts_are_private(self):
        for backend, stage in self.stages.items():
            with self.subTest(backend=backend):
                dependencies = stage / "third_party"
                gemmul8 = dependencies / "GEMMul8"
                if (ROOT / "source" / "third_party" / "GEMMul8" / "Makefile").exists():
                    self.assertTrue(gemmul8.is_dir())
                    self.assertFalse(gemmul8.is_symlink())
                    self.assertNotEqual(gemmul8.resolve(), ROOT / "source" / "third_party" / "GEMMul8")
                    self.assertFalse(list(gemmul8.rglob("*.o")))
                    self.assertFalse(list(gemmul8.rglob("*.a")))
                    self.assertFalse(list(gemmul8.rglob("*.mod")))
                    upstream = ROOT / "source" / "third_party" / "GEMMul8" / "lib" / "unified-test.a"
                    self.assertFalse(upstream.exists())
                    local = gemmul8 / "lib" / "unified-test.a"
                    local.parent.mkdir(exist_ok=True)
                    local.write_bytes(b"backend-private test library")
                    self.addCleanup(local.unlink, missing_ok=True)
                    self.assertFalse(upstream.exists())
                if backend == "HIP":
                    self.assertTrue((dependencies / "magma").is_dir())
                    self.assertFalse((dependencies / "magma").is_symlink())
                elif (dependencies / "fftw3").exists():
                    self.assertFalse((dependencies / "fftw3").is_symlink())
                # Pre-existing objects in source/ must never enter a fresh stage.
                self.assertFalse(list(stage.glob("*.o")))
                self.assertFalse(list(stage.glob("*.mod")))
                self.assertFalse((dependencies / "gpusolver2-install").exists())
                self.assertFalse((stage / "build").exists())
        cuda_gemm = self.stages["CUDA"] / "third_party" / "GEMMul8"
        hip_gemm = self.stages["HIP"] / "third_party" / "GEMMul8"
        self.assertNotEqual(cuda_gemm.resolve(), hip_gemm.resolve())

    def test_hip_regression_runners_find_default_input_templates(self):
        native = self.builds["HIP"] / "native"
        work = native / "work"
        self.assertTrue(work.is_dir(), "HIP regression runners need native/work inputs")
        self.assertFalse(work.is_symlink(), "regression outputs must remain build-private")
        for directory in ("input_example", "large_example", "large2_example", "large3_example"):
            with self.subTest(suite=directory):
                self.assertTrue((work / directory).is_symlink())
                self.assertEqual((work / directory).resolve(), ROOT / "work" / directory)
        # These are the default templates used by the copied band-spin,
        # multi-GPU Cluster and multi-GPU Band regression runners.
        templates = ("input_example/Crys-MnO.dat", "large_example/Pt63.dat",
                     "large_example/B2C62_Band.dat")
        original = {}
        for name in templates:
            with self.subTest(template=name):
                template = work / name
                self.assertTrue(template.is_file(), f"missing default regression template: {name}")
                self.assertEqual(template.resolve(), ROOT / "work" / name)
                original[name] = digest(template)
        # Exercise the real parsers and input preparation without MPI or GPUs.
        # Python is only copied as an executable placeholder, never launched.
        for runner in ("run_mgpu_cluster_regression.py", "run_mgpu_band_regression.py"):
            with self.subTest(runner=runner):
                output = self.workspace / Path(runner).stem
                completed = subprocess.run(
                    [sys.executable, str(native / "tests" / runner),
                     "--baseline", sys.executable, "--binary", sys.executable,
                     "--output", str(output), "--prepare-only",
                     "--cases", "col_off", "--backends", "off"],
                    capture_output=True, text=True, check=False,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                self.assertEqual(completed.returncode, 0, completed.stderr)
                manifest = json.loads((output / "command.json").read_text())
                self.assertTrue(manifest["prepared_only"])
                self.assertEqual(len(manifest["plans"]), 2)
                for plan in manifest["plans"]:
                    directory = Path(plan["directory"])
                    self.assertIn(output, directory.parents)
                    self.assertTrue((directory / "input.dat").is_file())
        for name, expected in original.items():
            self.assertEqual(digest(work / name), expected, f"modified shared input: {name}")

    def test_preparation_is_incremental_and_rejects_backend_switch(self):
        for backend, stage in self.stages.items():
            with self.subTest(backend=backend):
                output = stage / "unified-test.o"
                output.write_bytes(b"keep an existing native build result")
                self.addCleanup(output.unlink, missing_ok=True)
                before = output.stat().st_mtime_ns
                source_inode = (stage / "openmx.c").lstat().st_ino
                repeated = self.prepare(self.builds[backend], backend)
                self.assertEqual(repeated.returncode, 0, repeated.stderr)
                self.assertEqual(output.stat().st_mtime_ns, before)
                self.assertEqual((stage / "openmx.c").lstat().st_ino, source_inode)
                other = "HIP" if backend == "CUDA" else "CUDA"
                rejected = self.prepare(self.builds[backend], other)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("another backend", rejected.stderr)
                identity = json.loads((self.builds[backend] / ".openmx-backend.json").read_text())
                self.assertEqual(identity["backend"], backend)
                self.assertEqual(output.stat().st_mtime_ns, before)

    def test_selected_sources_require_only_this_repository(self):
        for backend, build in self.builds.items():
            record = json.loads((build / "selected-sources.json").read_text())
            self.assertEqual(record["backend"], backend)
            for relative in record["sources"].values():
                with self.subTest(backend=backend, source=relative):
                    self.assertFalse(Path(relative).is_absolute())
                    self.assertIn(ROOT, (ROOT / relative).resolve().parents)
                    self.assertTrue((ROOT / relative).is_file())
            self.assertNotIn("openmx4.0_amdgpu", (build / "selected-sources.json").read_text())

    def test_preparation_migrates_removed_hip_overrides(self):
        build = self.workspace / "hip-migration"
        self.assertEqual(self.prepare(build, "HIP").returncode, 0)
        stage = build / "native" / "source"
        name = "openmx_gpu_task.c"
        old = "source/backends/hip/src/" + name
        self.assertFalse((ROOT / old).exists())
        (stage / name).unlink()
        (stage / name).symlink_to(ROOT / old)
        record = build / "selected-sources.json"
        previous = json.loads(record.read_text())
        previous["sources"][name] = old
        record.write_text(json.dumps(previous))
        artifact = stage / "keep-existing.o"
        artifact.write_bytes(b"native build output")
        completed = self.prepare(build, "HIP")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((stage / name).resolve(), ROOT / "source" / name)
        self.assertEqual(artifact.read_bytes(), b"native build output")

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native rule dry runs")
    def test_kugui_core_selects_site_gcc_and_honors_override(self):
        environment = dict(os.environ)
        for name in ("OPENMX_CORE_CXX", "OPENMX_CORE_LIBRARY", "OPENMX_CORE_BACKEND"):
            environment.pop(name, None)
        for override in (False, True):
            with self.subTest(override=override):
                compiler = "/other/gcc/bin/g++" if override else "/site/gcc/12.2.0/bin/g++"
                command = ["make", "--no-print-directory", "-n", "-C", str(self.stages["CUDA"]),
                           "-f", str(ROOT / "source/Makefile.kugui"),
                           "KUGUI_GCC_ROOT=/site/gcc/12.2.0", "openmx-core"]
                if override:
                    command.append("OPENMX_CORE_CXX=" + compiler)
                completed = subprocess.run(command, capture_output=True, text=True,
                                           check=False, env=environment)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn('-DCMAKE_CXX_COMPILER="' + compiler + '"', completed.stdout)
                self.assertIn("-DOPENMX_CORE_BACKEND=SERIAL", completed.stdout)

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native rule dry runs")
    def test_pegasus_uses_shared_core_and_honors_overrides(self):
        environment = dict(os.environ)
        for name in ("OPENMX_CORE_CXX", "OPENMX_CORE_LIBRARY", "OPENMX_CORE_BACKEND"):
            environment.pop(name, None)
        for override in (False, True):
            with self.subTest(override=override):
                compiler = "/other/gcc/bin/g++" if override else "/usr/bin/g++-12"
                command = ["make", "--no-print-directory", "-n", "-C", str(self.stages["CUDA"]),
                           "-f", str(ROOT / "source/Makefile.pegasus"), "openmx-core"]
                if override:
                    command.append("OPENMX_CORE_CXX=" + compiler)
                completed = subprocess.run(command, capture_output=True, text=True,
                                           check=False, env=environment)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn('-DCMAKE_CXX_COMPILER="' + compiler + '"', completed.stdout)
                self.assertIn("-DOPENMX_CORE_BACKEND=SERIAL", completed.stdout)

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native rule dry runs")
    def test_native_core_uses_callers_cmake_and_honors_override(self):
        # CUDA replaces PATH for its compiler recipes. A CMake selected from
        # the caller's PATH must still configure/build the core afterward.
        # Execute the real rule with stand-ins; no compiler or download needed.
        tools = self.workspace / "cmake tools"
        tools.mkdir()
        log = tools / "calls.log"
        for name in ("cmake", "explicit-cmake"):
            executable = tools / name
            executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$0\" \"$1\" >> " +
                                  shlex.quote(str(log)) + "\n")
            executable.chmod(0o755)
        environment = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]}
        for name in ("OPENMX_CORE_CMAKE", "OPENMX_CORE_LIBRARY", "MAGMA_CMAKE"):
            environment.pop(name, None)
        make = shutil.which("make")
        for backend, stage in self.stages.items():
            for override in (False, True):
                with self.subTest(backend=backend, override=override):
                    log.unlink(missing_ok=True)
                    selected = tools / ("explicit-cmake" if override else "cmake")
                    command = [make, "--no-print-directory", "-C", str(stage), "openmx-core"]
                    if override:
                        command.append("OPENMX_CORE_CMAKE=" + str(selected))
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               check=False, env=environment)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertEqual(log.read_text().splitlines(),
                                     [str(selected), "-S", str(selected), "--build"])

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native rule dry runs")
    def test_explicit_cuda_architecture_does_not_query_a_gpu(self):
        tools = self.workspace / "gpu-unavailable"
        tools.mkdir(exist_ok=True)
        marker = tools / "queried"
        smi = tools / "nvidia-smi"
        smi.write_text("#!/bin/sh\ntouch " + shlex.quote(str(marker)) + "\nexit 99\n")
        smi.chmod(0o755)
        command = ["make", "--no-print-directory", "-n", "-C", str(self.stages["CUDA"]),
                   "GEMMUL8_GPU_ARCH=80", "WITH_MAGMA=0",
                   "PATH=" + str(tools) + os.pathsep + os.environ["PATH"],
                   "openmx.o", "openmx_gpu_dm.o"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertFalse(marker.exists(), "An explicit CUDA target must not query GPU hardware")
        self.assertIn("-gpu=cc80", completed.stdout)
        self.assertIn("compute_80,code=sm_80", completed.stdout)

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native rule dry runs")
    def test_native_object_recipes_preserve_each_backend_toolchain(self):
        for backend, stage in self.stages.items():
            with self.subTest(backend=backend):
                architecture = "GEMMUL8_GPU_ARCH=120" if backend == "CUDA" else "ARCH=gfx942"
                command = ["make", "--no-print-directory", "-n", "-C", str(stage), architecture,
                           "openmx.o", "openmx_gpu_dm.o", "Set_Density_Grid_GPU.o"]
                completed = subprocess.run(command, capture_output=True, text=True, check=False,
                                           env={**os.environ, "LC_ALL": "C"})
                self.assertEqual(completed.returncode, 0, completed.stderr)
                recipes = completed.stdout
                self.assertIn("-c openmx.c", recipes)
                self.assertIn("openmx_gpu_dm.cu", recipes)
                self.assertNotIn("openmx4.0_amdgpu", recipes)
                if backend == "CUDA":
                    self.assertIn("-acc", recipes)
                    self.assertIn("nvcc", recipes)
                    self.assertIn("compute_120", recipes)
                    self.assertNotIn("hipcc", recipes)
                else:
                    self.assertIn("--offload-arch=gfx942", recipes)
                    self.assertIn("hipcc", recipes)
                    self.assertIn("__HIP_PLATFORM_AMD__", recipes)
                    self.assertNotIn(" -acc ", recipes)
                self.assertFalse(list(stage.glob("*.o")), "make -n unexpectedly compiled an object")

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native dependency inspection")
    def test_registry_ids_invalidate_shared_hip_objects(self):
        stage = self.stages["HIP"]
        objects = ("openmx.o", "DIIS_Mixing_DM.o", "Kerker_Mixing_Rhok.o", "Total_Energy.o")
        completed = subprocess.run(
            ["make", "--no-print-directory", "-np", "-C", str(stage), "ARCH=gfx942", *objects],
            capture_output=True, text=True, check=False, env={**os.environ, "LC_ALL": "C"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for obj in objects:
            with self.subTest(object=obj):
                row = next(line for line in completed.stdout.splitlines() if line.startswith(obj + ":"))
                self.assertIn("openmx_env.h", row.split())
                self.assertIn("openmx_env_registry.h", row.split())
        self.assertFalse(list(stage.glob("*.o")), "make -np unexpectedly compiled an object")

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for native dependency inspection")
    def test_shared_inline_headers_invalidate_native_objects(self):
        objects = ("Total_Energy.o", "Stress.o", "Cluster_DFT_Col_CWF.o",
                   "Cluster_DFT_NonCol_CWF.o", "Band_DFT_Col_CWF.o", "Band_DFT_NonCol_CWF.o")
        for backend, stage in self.stages.items():
            architecture = "GEMMUL8_GPU_ARCH=120" if backend == "CUDA" else "ARCH=gfx942"
            completed = subprocess.run(
                ["make", "--no-print-directory", "-np", "-C", str(stage), architecture, "FFTW3_LIB=", *objects],
                capture_output=True, text=True, check=False, env={**os.environ, "LC_ALL": "C"})
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for obj in objects:
                with self.subTest(backend=backend, object=obj):
                    row = next(line for line in completed.stdout.splitlines() if line.startswith(obj + ":"))
                    # Existing public ABI dependencies use canonical absolute paths.
                    self.assertTrue(any(p.endswith("/core/include/openmx_core_neutral_grid_metadata.h")
                                        for p in row.split()), row)
                    helper = "openmx_cwf_workspace.h" if "CWF" in obj else "openmx_neutral_atom.h"
                    self.assertIn(helper, row.split())
            self.assertFalse(list(stage.glob("*.o")), "make -np unexpectedly compiled an object")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Exercise the real CMake driver with a tiny core and a fake native make."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc"), "CMake and a C compiler are required")
class CMakeDriverTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="openmx cmake driver ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.build = self.root / "build"
        for name in ("cmake", "tools", "core", "modules", "source/third_party/dist", "DFT_DATA19"):
            (self.repo / name).mkdir(parents=True, exist_ok=True)
        for name in ("CMakeLists.txt", "cmake/OpenMXGPUArchitecture.cmake", "cmake/OpenMXBuildType.cmake",
                     "cmake/OpenMXCPUOptions.cmake", "cmake/OpenMXMPI.cmake", "cmake/OpenMXHIPOptions.cmake",
                     "tools/prepare_build.py", "tools/copy_openmx.py"):
            shutil.copy2(ROOT / name, self.repo / name)
        (self.repo / "core/core.c").write_text("int fixture(void) { return 0; }\n")
        (self.repo / "core/CMakeLists.txt").write_text(
            'project(FixtureCore LANGUAGES C)\n'
            'set(OPENMX_CORE_BACKEND SERIAL CACHE STRING "")\n'
            'add_library(openmx_core SHARED core.c)\n')
        (self.repo / "source/Makefile").write_text("# Native make is substituted by the fixture.\n")
        self.payload = self.root / "payload"
        self.payload.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.native_make = self.root / "native-make"
        self.native_make.write_text(
            "#!" + sys.executable + "\nimport json, os, pathlib, sys\n"
            "os.chdir(sys.argv[sys.argv.index('-C')+1])\n"
            "pathlib.Path('native-args.json').write_text(json.dumps(sys.argv[1:]))\n"
            "if os.environ.get('OPENMX_TEST_FAIL'): sys.exit(23)\n"
            "p=pathlib.Path('openmx');p.write_bytes(pathlib.Path(os.environ['OPENMX_TEST_PAYLOAD']).read_bytes());p.chmod(0o755)\n")
        self.native_make.chmod(0o755)
        self.gpu_tools = self.root / "gpu tools"
        self.gpu_tools.mkdir()
        self.gpu_queries = self.gpu_tools / "queries.jsonl"
        self.nvidia_smi = self.gpu_tools / "nvidia-smi"
        self.mock_nvidia_smi(result=99, error="GPU unavailable")
        self.env = {**os.environ, "OPENMX_TEST_PAYLOAD": str(self.payload), "CUDA_VISIBLE_DEVICES": "",
                    "PATH": str(self.gpu_tools) + os.pathsep + os.environ["PATH"]}

    def mock_nvidia_smi(self, output="", result=0, error=""):
        self.nvidia_smi.write_text(
            "#!" + sys.executable + "\nimport json, pathlib, sys\n"
            "with pathlib.Path(" + repr(str(self.gpu_queries)) + ").open('a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.stdout.write(" + repr(output) + ")\n"
            "sys.stderr.write(" + repr(error) + ")\n"
            "sys.exit(" + str(result) + ")\n")
        self.nvidia_smi.chmod(0o755)

    def command(self, args, success=True, env=None):
        result = subprocess.run(args, capture_output=True, text=True, env=env or self.env)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def configure(self, *options, success=True):
        return self.command(["cmake", "-S", str(self.repo), "-B", str(self.build), "-DBUILD_TESTING=OFF",
                             "-DCMAKE_C_COMPILER=" + shutil.which("cc"),
                             "-DOPENMX_MAKE_EXECUTABLE=" + str(self.native_make), *options], success=success)

    def build_application(self, **kwargs):
        return self.command(["cmake", "--build", str(self.build), "--target", "openmx"], **kwargs)

    def test_hip_selectors_reject_unknown_values_before_sdk_detection(self):
        for option, value in (("MPI", "MPICH"), ("LINALG", "ILP64")):
            with self.subTest(option=option):
                result = self.configure("-DOPENMX_GPU_BACKEND=HIP",
                                        "-DOPENMX_HIP_MPI=OPENMPI", "-DOPENMX_HIP_LINALG=AOCL",
                                        f"-DOPENMX_HIP_{option}={value}", success=False)
                self.assertIn(f"OPENMX_HIP_{option} must be", result.stdout + result.stderr)
                self.assertFalse(self.gpu_queries.exists())

    def test_hip_rejects_native_toolchain_overrides(self):
        result = self.configure("-DOPENMX_GPU_BACKEND=HIP",
                                "-DOPENMX_MAKE_OPTIONS=MPICC=/different/mpi/mpicc", success=False)
        self.assertIn("MPICC is managed by the HIP CMake toolchain", result.stdout + result.stderr)
        self.assertFalse(self.gpu_queries.exists())

    def test_hip_rejects_conflicting_legacy_prefix(self):
        result = self.configure("-DOPENMX_GPU_BACKEND=HIP", "-DOPENMX_HIP_MPI_ROOT=/selected/mpi",
                                "-DOPENMX_MAKE_OPTIONS=MPI_HOME=/different/mpi", success=False)
        self.assertIn("conflicts with OPENMX_HIP_MPI_ROOT", result.stdout + result.stderr)
        self.assertFalse(self.gpu_queries.exists())

    def test_default_release_and_explicit_debug_configuration(self):
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        cache = self.build / "CMakeCache.txt"
        self.assertIn("CMAKE_BUILD_TYPE:STRING=Release", cache.read_text())
        self.command(["cmake", "--build", str(self.build), "--target", "openmx_core"])
        commands = self.build / "compile_commands.json"
        self.assertIn("-O3", json.loads(commands.read_text())[0]["command"])
        self.configure("-DCMAKE_BUILD_TYPE=Debug")
        self.assertIn("CMAKE_BUILD_TYPE:STRING=Debug", cache.read_text())
        self.command(["cmake", "--build", str(self.build), "--target", "openmx_core"])
        self.assertNotIn("-O3", json.loads(commands.read_text())[0]["command"])
        self.configure("-DCMAKE_BUILD_TYPE=")
        self.assertIn("CMAKE_BUILD_TYPE:STRING=Release", cache.read_text())

    def test_cpu_selection_does_not_query_gpu_hardware(self):
        self.configure('-DOPENMX_GPU_BACKEND=CPU', '-DOPENMX_BUILD_APPLICATION=OFF')
        self.command(['cmake', '--build', str(self.build), '--target', 'openmx_core'])
        self.assertFalse(self.gpu_queries.exists())

    @unittest.skipUnless(shutil.which("ninja"), "Ninja required")
    def test_ninja_application_target_and_output_can_share_a_name(self):
        self.configure("-G", "Ninja", "-DOPENMX_GPU_ARCH=sm_80")
        self.build_application()
        self.assertEqual((self.build / "openmx").read_bytes(), self.payload.read_bytes())
        self.payload.write_bytes(b"#!/bin/sh\nexit 2\n")
        self.build_application()
        self.assertEqual((self.build / "openmx").read_bytes(), self.payload.read_bytes())

    def test_cpu_application_rejects_device_core_before_sdk_detection(self):
        failed = self.configure('-DOPENMX_GPU_BACKEND=CPU', '-DOPENMX_CORE_BACKEND=CUDA', success=False)
        self.assertIn('CPU application requires OPENMX_CORE_BACKEND=SERIAL or OPENMP', failed.stderr)
        self.assertFalse(self.gpu_queries.exists())

    def test_cpu_toolchain_selectors_reject_unknown_values_before_detection(self):
        for option, value in [('COMPILER', 'NVCC'), ('MPI', 'MPICH'), ('LINALG', 'ACCELERATE')]:
            with self.subTest(option=option):
                failed = self.configure('-DOPENMX_GPU_BACKEND=CPU',
                                        '-DOPENMX_CPU_' + option + '=' + value, success=False)
                self.assertIn('OPENMX_CPU_' + option + ' must be', failed.stderr)
                self.assertFalse(self.gpu_queries.exists())
                # Each independent invalid selection starts with a clean cache.
                shutil.rmtree(self.build)

    def test_success_copy_opt_out_custom_directory_and_failed_build(self):
        self.configure("-DOPENMX_GPU_ARCH=sm_80")
        self.build_application()
        self.assertFalse(self.gpu_queries.exists(), "An explicit CUDA target must not query a GPU")
        work = self.repo / "work/openmx"
        self.assertEqual(work.read_bytes(), self.payload.read_bytes())
        self.assertTrue(os.access(work, os.X_OK))
        args = json.loads((self.build / "native/source/native-args.json").read_text())
        self.assertIn("GEMMUL8_GPU_ARCH=80", args)
        self.payload.write_bytes(b"#!/bin/sh\nexit 2\n")
        self.configure("-DOPENMX_COPY_TO_WORK=OFF")
        self.build_application()
        self.assertNotEqual(work.read_bytes(), self.payload.read_bytes())
        self.configure("-DOPENMX_COPY_TO_WORK=ON")
        old = work.read_bytes()
        self.build_application(success=False, env={**self.env, "OPENMX_TEST_FAIL": "1"})
        self.assertEqual(work.read_bytes(), old)
        custom = self.root / "custom work"
        self.configure("-DOPENMX_WORK_DIR=" + str(custom))
        self.build_application()
        self.assertEqual((custom / "openmx").read_bytes(), self.payload.read_bytes())
        self.assertEqual(work.read_bytes(), old)

    def architecture_script(self, code, success=True):
        script = self.root / "architecture.cmake"
        module = ROOT / "cmake/OpenMXGPUArchitecture.cmake"
        # Isolate program discovery so tests never depend on a host GPU or
        # accidentally find the real nvidia-smi in the missing-command case.
        script.write_text('cmake_minimum_required(VERSION 3.25.2)\n'
                          'set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n'
                          'set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n'
                          'set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)\n'
                          'set(CMAKE_PROGRAM_PATH [=[' + str(self.gpu_tools) + ']=])\n'
                          'include([=[' + str(module) + ']=])\n' + code)
        return self.command(["cmake", "-P", str(script)], success=success)

    def test_architecture_reaches_kokkos_and_can_be_changed_without_a_gpu(self):
        self.architecture_script('''
set(OPENMX_GPU_ARCH sm_80)
openmx_resolve_cuda_architecture()
openmx_select_kokkos_cuda_architecture("${OPENMX_CUDA_ARCH}")
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "80" OR NOT Kokkos_ARCH_AMPERE80)
  message(FATAL_ERROR "The explicit target was lost")
endif()
set(OPENMX_GPU_ARCH sm_90)
openmx_resolve_cuda_architecture()
openmx_select_kokkos_cuda_architecture("${OPENMX_CUDA_ARCH}")
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "90" OR NOT Kokkos_ARCH_HOPPER90 OR Kokkos_ARCH_AMPERE80)
  message(FATAL_ERROR "Changing the target left a conflicting cache entry")
endif()
''')
        self.assertFalse(self.gpu_queries.exists())

    def test_standard_cmake_architecture_and_invalid_combinations(self):
        self.architecture_script('''
set(OPENMX_GPU_ARCH sm_120)
set(CMAKE_CUDA_ARCHITECTURES 120)
openmx_resolve_cuda_architecture()
openmx_select_kokkos_cuda_architecture("${OPENMX_CUDA_ARCH}")
if(NOT OPENMX_CUDA_ARCH STREQUAL "120" OR NOT Kokkos_ARCH_BLACKWELL120)
  message(FATAL_ERROR "CMake's explicit target was lost")
endif()
''')
        for code in ('set(OPENMX_GPU_ARCH "80;90")',
                     'set(OPENMX_GPU_ARCH all)',
                     'set(OPENMX_GPU_ARCH OFF)',
                     'set(OPENMX_GPU_ARCH sm_80)\nset(CMAKE_CUDA_ARCHITECTURES all)',
                     'set(OPENMX_GPU_ARCH sm_80)\nset(CMAKE_CUDA_ARCHITECTURES 90)'):
            self.architecture_script(code + '\nopenmx_resolve_cuda_architecture()\n', success=False)

    def test_automatic_architecture_reaches_native_build_and_refreshes_the_cache(self):
        self.mock_nvidia_smi(output="12.0\n")
        configured = self.configure()
        self.assertIn("auto-detected with nvidia-smi: sm_120", configured.stdout)
        self.build_application()
        args_file = self.build / "native/source/native-args.json"
        self.assertIn("GEMMUL8_GPU_ARCH=120", json.loads(args_file.read_text()))
        self.mock_nvidia_smi(output="9.0\n")
        self.configure()
        self.build_application()
        self.assertIn("GEMMUL8_GPU_ARCH=90", json.loads(args_file.read_text()))
        self.assertEqual([json.loads(line) for line in self.gpu_queries.read_text().splitlines()],
                         [["--query-gpu=compute_cap", "--format=csv,noheader"]] * 2)
        self.mock_nvidia_smi(result=9, error="No devices were found")
        failed = self.configure(success=False)
        self.assertIn("nvidia-smi failed", failed.stderr)
        self.assertIn("-DOPENMX_GPU_ARCH=sm_80", failed.stderr)

    def test_automatic_architecture_reaches_kokkos(self):
        self.mock_nvidia_smi(output="N/A\n  12.0\r\n8.0\n")
        for requested in ("", "set(OPENMX_GPU_ARCH auto)"):
            with self.subTest(requested=requested):
                self.architecture_script(requested + '''
openmx_resolve_cuda_architecture()
openmx_select_kokkos_cuda_architecture("${OPENMX_CUDA_ARCH}")
if(NOT OPENMX_CUDA_ARCH STREQUAL "120" OR NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120" OR NOT Kokkos_ARCH_BLACKWELL120)
  message(FATAL_ERROR "The automatic target did not reach CUDA and Kokkos")
endif()
''')

    def test_missing_nvidia_smi_is_an_error_even_with_a_cached_target(self):
        self.nvidia_smi.unlink()
        for cache in ("", "set(CMAKE_CUDA_ARCHITECTURES 80)\nset(OPENMX_MANAGED_CUDA_ARCH 80)"):
            with self.subTest(cache=cache):
                failed = self.architecture_script(cache + "\nopenmx_resolve_cuda_architecture()\n", success=False)
                self.assertIn("nvidia-smi was not found", failed.stderr)
                self.assertIn("-DOPENMX_GPU_ARCH=sm_80", failed.stderr)

    def test_unusable_nvidia_smi_is_an_error_even_with_a_cached_target(self):
        cases = (("", 9, "No devices were found", "nvidia-smi failed"),
                 ("12.0\n", 9, "Driver failure", "nvidia-smi failed"),
                 ("", 0, "", "no GPU with a valid compute capability"),
                 ("N/A\n", 0, "", "no GPU with a valid compute capability"),
                 ("No devices were found\n", 0, "", "no GPU with a valid compute capability"))
        for output, result, error, expected in cases:
            with self.subTest(output=output, result=result):
                self.mock_nvidia_smi(output=output, result=result, error=error)
                failed = self.architecture_script('''
set(CMAKE_CUDA_ARCHITECTURES 120)
set(OPENMX_MANAGED_CUDA_ARCH 120)
openmx_resolve_cuda_architecture()
''', success=False)
                self.assertIn(expected, " ".join(failed.stderr.split()))
                self.assertIn("-DOPENMX_GPU_ARCH=sm_80", failed.stderr)

    def test_host_core_only_build_does_not_query_a_gpu(self):
        self.configure("-DOPENMX_BUILD_APPLICATION=OFF")
        self.command(["cmake", "--build", str(self.build)])
        self.assertFalse(self.gpu_queries.exists())

    def test_install_uses_the_configured_build_directory_and_prefix(self):
        self.build = self.root / "build/cuda"
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF")
        prefix = self.root / "installed OpenMX"
        install = ["cmake", "--install", str(self.build), "--prefix", str(prefix)]
        # Installation does not build the application; the normal build must
        # finish first. The parent directory is not a CMake build directory.
        self.command(install, success=False)
        self.build_application()
        self.command(["cmake", "--install", str(self.build.parent), "--prefix", str(prefix)], success=False)
        self.command(install)
        executable = prefix / "bin/openmx"
        self.assertEqual(executable.read_bytes(), self.payload.read_bytes())
        self.assertTrue(os.access(executable, os.X_OK))
        self.assertEqual((prefix / "lib/libopenmx_core.so").read_bytes(),
                         (self.build / "core/libopenmx_core.so").read_bytes())
        manifest = (self.build / "install_manifest.txt").read_text().splitlines()
        self.assertIn(str(executable), manifest)
        self.assertIn(str(prefix / "lib/libopenmx_core.so"), manifest)

    def assert_install_layout(self, prefix, program):
        self.assertEqual(program.read_bytes(), self.payload.read_bytes())
        self.assertTrue(os.access(program, os.X_OK))
        self.assertEqual((prefix / "lib/libopenmx_core.so").read_bytes(),
                         (self.build / "core/libopenmx_core.so").read_bytes())
        manifest = (self.build / "install_manifest.txt").read_text().splitlines()
        self.assertIn(str(program), manifest)
        self.assertIn(str(prefix / "lib/libopenmx_core.so"), manifest)

    def test_default_install_places_the_binary_directly_in_work(self):
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF")
        self.build_application()
        work = self.repo / "work"
        self.assertFalse(work.exists(), "The build must not mask a missing install step")
        self.command(["cmake", "--install", str(self.build)])
        self.assert_install_layout(work, work / "openmx")
        self.assertFalse((work / "bin/openmx").exists())

    def test_configured_install_prefix_is_honored_and_can_be_reset(self):
        prefix = self.root / "configured prefix"
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF",
                       "-DCMAKE_INSTALL_PREFIX=" + str(prefix))
        self.build_application()
        self.command(["cmake", "--install", str(self.build)])
        self.assert_install_layout(prefix, prefix / "bin/openmx")
        self.assertFalse((self.repo / "work").exists())
        # Resetting a previous /usr/local or site-specific cache value must
        # activate the new local default without writing to that old prefix.
        self.configure("-UCMAKE_INSTALL_PREFIX")
        self.build_application()
        self.command(["cmake", "--install", str(self.build)])
        work = self.repo / "work"
        self.assert_install_layout(work, work / "openmx")

    def test_default_install_tracks_the_selected_work_directory(self):
        for directory in ("first work", "second work"):
            with self.subTest(directory=directory):
                work = self.root / directory
                self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF",
                               "-DOPENMX_WORK_DIR=" + str(work))
                self.build_application()
                self.assertFalse(work.exists())
                self.command(["cmake", "--install", str(self.build)])
                self.assert_install_layout(work, work / "openmx")

    def test_default_install_honors_destdir_staging(self):
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF")
        self.build_application()
        work = self.repo / "work"
        staging = self.root / "packaging root"
        self.command(["cmake", "--install", str(self.build)], env={**self.env, "DESTDIR": str(staging)})
        installed = Path(str(staging) + str(work))
        self.assertEqual((installed / "openmx").read_bytes(), self.payload.read_bytes())
        self.assertEqual((installed / "lib/libopenmx_core.so").read_bytes(),
                         (self.build / "core/libopenmx_core.so").read_bytes())
        self.assertFalse(work.exists(), "Staging must not write outside DESTDIR")
        self.assertIn(str(work / "openmx"), (self.build / "install_manifest.txt").read_text().splitlines())

    @unittest.skipUnless(shutil.which("make"), "GNU make is required for the native link rule")
    def test_installed_program_loads_the_core_from_both_layouts(self):
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF")
        self.build_application()
        # The real native fragment passes its absolute library path to the
        # linker. Give that build-only path no spaces, then remove it to make
        # the loader rely on the installed relative paths in each layout.
        with tempfile.TemporaryDirectory(prefix="openmx-install-link-") as temporary:
            link = Path(temporary)
            core = link / "libopenmx_core.so"
            shutil.copy2(self.build / "core/libopenmx_core.so", core)
            (link / "main.c").write_text("int fixture(void); int main(void) { return fixture(); }\n")
            (link / "Makefile").write_text(
                "OPENMX_CORE_LIBRARY := " + str(core) + "\n"
                "OBJS := unused.o\n"
                "include " + str(ROOT / "source/openmx_core.mk") + "\n"
                "payload: main.c\n\t" + shutil.which("cc") + " -o $@ $< $(LIB)\n")
            self.command([shutil.which("make"), "-C", str(link), "payload"])
            self.payload.write_bytes((link / "payload").read_bytes())
            self.build_application()
            core.unlink()
            for prefix in (None, self.root / "relocated prefix"):
                with self.subTest(prefix=prefix):
                    command = ["cmake", "--install", str(self.build)]
                    if prefix is None:
                        executable = self.repo / "work/openmx"
                    else:
                        command.extend(["--prefix", str(prefix)])
                        executable = prefix / "bin/openmx"
                    self.command(command)
                    self.command([str(executable)])

    @unittest.skipUnless(Path("/bin/sleep").is_file() and Path("/bin/true").is_file(), "POSIX executables are required")
    def test_default_install_replaces_a_running_executable(self):
        self.configure("-DOPENMX_GPU_ARCH=sm_80", "-DOPENMX_COPY_TO_WORK=OFF")
        self.payload.write_bytes(Path("/bin/true").read_bytes())
        self.build_application()
        destination = self.repo / "work/openmx"
        destination.parent.mkdir()
        shutil.copy2("/bin/sleep", destination)
        process = subprocess.Popen([str(destination), "30"])
        try:
            self.command(["cmake", "--install", str(self.build)])
            self.assertIsNone(process.poll())
            self.assert_install_layout(destination.parent, destination)
            self.command([str(destination)])
        finally:
            process.terminate()
            process.wait(timeout=5)

    @unittest.skipUnless(Path("/bin/sleep").is_file() and Path("/bin/true").is_file(), "POSIX executables are required")
    def test_copy_replaces_a_running_executable_atomically(self):
        destination = self.root / "running openmx"
        shutil.copy2("/bin/sleep", destination)
        process = subprocess.Popen([str(destination), "30"])
        try:
            self.command([sys.executable, str(ROOT / "tools/copy_openmx.py"), "/bin/true", str(destination)])
            self.assertIsNone(process.poll())
            self.assertEqual(destination.read_bytes(), Path("/bin/true").read_bytes())
            self.command([str(destination)])
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

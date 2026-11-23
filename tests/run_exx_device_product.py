#!/usr/bin/env python3
"""Run the device products of the exchange build against a model of the device libraries.

source/Exact_Exchange_Device.c is compiled as in the CUDA build, with the
headers of tests/fixtures/exx_device_sdk in place of the SDK, and linked with
tests/exx_device_product_harness.c, which defines the calls of the model.
No GPU and no SDK are needed.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory(prefix='openmx-exx-device-') as directory:
    binary = Path(directory)/'harness'
    command = shlex.split(os.environ.get('CC', 'cc')) + [
        '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
        '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
        '-DOPENMX_GPU_BACKEND=1',
        '-I'+str(ROOT/'tests/fixtures/exx_device_sdk'), '-I'+str(ROOT/'source'),
        str(ROOT/'source/Exact_Exchange_Device.c'), str(ROOT/'tests/exx_device_product_harness.c'),
        '-lm', '-o', str(binary)]
    subprocess.run(command, check=True)
    subprocess.run([str(binary)], check=True, timeout=120)

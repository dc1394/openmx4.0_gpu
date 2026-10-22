"""Shared adapters for native Force extraction fixtures.

Archive/stream arithmetic fixtures model the SDK and context boundary; recovery
fixtures link the real C++ packet owner. Separate lifetime tests exercise native
resource registration and teardown against the actual C++ owner.
"""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

CONTEXTS = {
    "Force4BGpuContext": ("F4B_gpu", "fixture_gpu", [
        "Force4B_GpuBegin", "Force4B_GpuEnd", "Force4B_GpuCase2Begin",
        "Force4B_GpuArchiveHalo", "Force4B_GpuArchiveCase2",
        "Force4B_GpuCase1Run", "Force4B_GpuCase2Run"]),
    "Force4BCase1Stream": ("F4B_stream1", "fixture_stream1", [
        "Force4B_GpuCase1StreamEnd", "Force4B_GpuCase1PlanAdd",
        "Force4B_GpuCase1StreamBegin", "Force4B_GpuCase1StreamSource",
        "Force4B_GpuCase1StreamAccumulate"]),
    "Force4BCase1Turns": ("F4B_turns1", "fixture_turns1", [
        "Force4B_GpuCase1TurnsEnd", "Force4B_GpuCase1TurnsAllocate",
        "Force4B_GpuCase1TurnsBegin", "Force4B_GpuCase1TurnsArchive",
        "Force4B_GpuCase1TurnsDevice", "Force4B_GpuCase1TurnsAccumulate",
        "Force4B_GpuCase1TurnsRun"]),
    "Force4BCase2Stream": ("F4B_stream", "fixture_stream2", [
        "Force4B_GpuCase2StreamEnd", "Force4B_GpuCase2StreamBegin",
        "Force4B_GpuCase2StreamRun", "Force4B_GpuCase2StreamAccumulate"]),
}


def explicit_fixture_contexts(harness):
    """Adapt fixture calls only; production signatures/calls remain verbatim."""
    for typ, (old, pointer, names) in CONTEXTS.items():
        harness = re.sub(r"static " + typ + " " + old + r" = [^\n]*;",
                         f"static {typ} *{pointer};\n#define {old} (*{pointer})", harness)
        for name in names:
            def arguments(match):
                rest = harness[match.end():]
                if re.match(r"(?:g\b|\*output\b|scratch\b|" + typ + r"\b|OpenMXSolverInvocation\b)", rest):
                    return match[0]
                prefix = "NULL, &" + pointer if name.endswith("Begin") and name != "Force4B_GpuCase2Begin" else pointer
                return match[0] + prefix + ("" if rest.startswith(")") else ", ")
            harness = re.sub(r"\b" + name + r"\(", arguments, harness)
    return harness


def model(hip=False):
    return ("\n#define FORCE_FIXTURE_HIP " + str(int(hip)) + "\n#include \"" +
            str(ROOT / "tests/fixtures/force_gpu_resources_model.h") + "\"\n")


def core_library_flags():
    """Link recovery arithmetic fixtures to the real C++ packet owner."""
    configured = os.environ.get("OPENMX_CORE_LIBRARY")
    candidates = [Path(configured)] if configured else [
        *(parent / "core/libopenmx_core.so" for parent in Path(sys.argv[0]).resolve().parents),
        *(ROOT / "build" / backend / "core/libopenmx_core.so" for backend in ("cuda", "hip")),
        ROOT / "build/core/libopenmx_core.so",
    ]
    library = next((path.resolve() for path in candidates if path.is_file()), None)
    if library is None:
        raise FileNotFoundError("Build openmx_core and set OPENMX_CORE_LIBRARY to its shared library")
    return [str(library), "-Wl,-rpath," + str(library.parent)]

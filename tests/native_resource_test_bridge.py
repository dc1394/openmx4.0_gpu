"""Isolate the numerical/SDK control flow in legacy host extraction harnesses.

These harnesses have no DFT context or real device. Lower only the ownership
adapter boundary to the SDK/heap stubs they already supply. The production
ownership adapters themselves are compiled and exercised by the independent
core-native-resource-* and core-native-adapter-* tests.
"""
import re


def calculation_state_declaration(source, alias):
    """Keep the actual POD types/layout in narrow SDK policy fixtures.

    Callers supply SDK typedefs. This copies declarations, not initialization
    or ownership behavior; real-core lifecycle tests cover those separately.
    """
    start = source.index("/* Persistent metadata")
    start = source.index("*/", start) + 2
    end = source.index(" " + alias + ";", start) + len(" " + alias + ";")
    return source[start:end]


def isolate_native_resources(source, backend="CUDA"):
    def mask(match):
        return "".join("\n" if char == "\n" else " " for char in match[0])

    clean = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', mask, source)
    sdk = {
        "GpuMalloc": "cudaMalloc" if backend == "CUDA" else "hipMalloc",
        "GpuFree": "cudaFree" if backend == "CUDA" else "hipFree",
        "HostFree": "cudaFreeHost" if backend == "CUDA" else "hipHostFree",
        "StreamCreate": "cudaStreamCreateWithFlags" if backend == "CUDA" else "hipStreamCreateWithFlags",
        "StreamDestroy": "cudaStreamDestroy" if backend == "CUDA" else "hipStreamDestroy",
        "BlasCreate": "cublasCreate" if backend == "CUDA" else "hipblasCreate",
        "BlasDestroy": "cublasDestroy" if backend == "CUDA" else "hipblasDestroy",
        "SolverCreate": "cusolverDnCreate" if backend == "CUDA" else "hipsolverDnCreate",
        "SolverDestroy": "cusolverDnDestroy" if backend == "CUDA" else "hipsolverDnDestroy",
        "AccMalloc": "acc_malloc", "AccFree": "acc_free",
    }
    items = []
    pattern = r"\b(OpenMX_Resource_(?:" + "|".join([*sdk, "HostAlloc"]) + r")|openmx_core_native_(?:malloc|realloc|free))\s*\("
    for match in re.finditer(pattern, clean):
        depth, end, start, args = 1, match.end(), match.end(), []
        while depth:
            char = clean[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 1:
                args.append(source[start:end].strip())
                start = end + 1
            end += 1
        args.append(source[start:end - 1].strip())
        name = match[1]
        if name == "openmx_core_native_malloc":
            _, count, size, zero = args
            expression = ("calloc(" + count + "," + size + ")") if zero == "1" else (
                "malloc(" + (size if count == "1" else "(" + count + ")*(" + size + ")") + ")")
        elif name.startswith("openmx_core_native_"):
            expression = name.removeprefix("openmx_core_native_") + "(" + ",".join(args[1:]) + ")"
        else:
            name = name.removeprefix("OpenMX_Resource_")
            if name == "HostAlloc":
                name = "cudaMallocHost" if backend == "CUDA" else "hipHostMalloc"
                if backend == "CUDA":
                    if args[-1] != "0":
                        name = "cudaHostAlloc"
                    else:
                        args = args[:-1]
            else:
                name = sdk[name]
            expression = name + "(" + ",".join(args[1:]) + ")"
        items.append((match.start(), end, expression))
    for start, end, expression in reversed(items):
        source = source[:start] + expression + source[end:]
    # An extracted solver has no calculation owner; retain its borrowed field.
    source = re.sub(r"(\bdev(?:->|\.)resources\s*=\s*)\w+_resources\s*;", r"\1NULL;", source)
    # These older harnesses use one integer-success checker for all their
    # stubbed SDK families. Production keeps the typed checkers.
    source = re.sub(r"\b(?:OpenMXBlasWait|OpenMXSolverWait)\b",
                    "wait_cudafunc" if backend == "CUDA" else "wait_hipfunc", source)
    return source

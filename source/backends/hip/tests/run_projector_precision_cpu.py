#!/usr/bin/env python3
"""Run the production VNA/force contraction loops on host test arrays.

OpenMP target pragmas are ignored by the host compiler.  Each test compares the
actual loop against a long-double reference and exercises both legacy FP32
product rounding and FP64 products, including local/halo rows and batches.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def block(text, start):
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


from run_force4b_stream_cpu import extract_function

force = (ROOT / "source/Force.c").read_text()
vna = (ROOT / "source/Set_ProExpn_VNA.c").read_text()
item = re.search(r"typedef struct \{[^{}]+\} Force4BGpuItem;", force).group()
functions = "\n".join(
    extract_function(force,name+"_Single").replace(name+"_Single",name)
    for name in ("Force4B_GpuCase1Kernel", "Force4B_GpuCase2Kernel")
)
functions += "\n" + block(vna, vna.index("static double SetPro_HVNA_element("))
functions += r"""
static void vna_kernel(int nitems_c, int num_proj, int num_rvna,
    const int *item_koff, const int *item_kn,
    const int *item_tnoC, const int *item_tnoH, const size_t *item_out,
    const float *flat, const float *halo, const size_t *krowA,
    const size_t *krowB, const int *krow_halo, const int *krow_ene,
    const double *ene, const int *l2p1, double *out)
{
    int out_item[6];
    const int fp32 = openmx_fp32_enabled();
    for (int p = 0; p < nitems_c; p++)
        for (int i = 0; i < item_tnoC[p] * item_tnoH[p]; i++)
            out_item[item_out[p] + i] = p;
    for (int oe = 0; oe < 6; oe++)
        out[oe] = SetPro_HVNA_element(oe, num_proj, num_rvna, flat, halo,
            out_item, item_koff, item_kn, item_tnoH, item_out, krowA, krowB,
            krow_halo, krow_ene, ene, l2p1, fp32);
}
"""

# Exercise the actual local/halo derivative contractions in both the force
# and stress implementations; their surrounding atom lookup does not affect
# the FP32/FP64 product policy.
derivative_calls = []
for filename in ("Force.c", "Stress.c"):
    source = (ROOT / "source" / filename).read_text()
    source = block(source, source.index("\nvoid dHVNA(") + 1)
    loops = []
    for match in re.finditer(r"for\s*\(l\s*=\s*0;\s*l\s*<\s*num_projectors;\s*l\+\+\)\s*\{", source):
        loop = block(source, match.start())
        if "fp32" in loop and "sumx" in loop:
            loops.append(loop)
    assert len(loops) == 4, f"{filename}: expected four derivative contractions"
    for index, loop in enumerate(loops):
        name = filename.removesuffix(".c").lower() + f"_dot_{index}"
        functions += f"\nstatic void {name}(float DS_VNA1[4][4][3][2][9], double out[3]) {{\n" + r"""
    const int fp32 = openmx_fp32_enabled(), num_projectors = 9;
    const int Mc_AN = 1, Mj_AN2 = 2, Matomnum = 2, k = 2, kl = 1, kl1 = 1, kl2 = 2, m = 0, n = 1;
    int l;
    double sumx = 0.0, sumy = 0.0, sumz = 0.0;
    (void)Mc_AN; (void)Mj_AN2; (void)Matomnum; (void)k; (void)kl; (void)kl1; (void)kl2;
""" + loop + "\n    out[0] = sumx; out[1] = sumy; out[2] = sumz;\n}\n"
        derivative_calls.append(f"    {name}(data, result); derivative_check({index}, legacy, data, result);")

HARNESS = r"""
#define P 9
#define ROWS 6
static const size_t stride = P * ROWS;
static float flat[4 * P * ROWS], halo[2 * P], c2[4 * P * ROWS];
static const size_t ka[] = {0, 4 * P, P}, kb[] = {2 * P, 0, 3 * P};
static const int is_halo[] = {0, 1, 0};
static const double weights[] = {0.3, -0.7, 0.6, 0.2, -0.4, 0.9};
static const Force4BGpuItem items[] = {{0, 0, 2, 2, 2}, {4, 2, 1, 1, 2}};

static long double product(float a, float b, int legacy)
{
    /* float multiplication is exact in long double, then rounded only
       for the reference corresponding to the legacy FP32 policy. */
    long double exact = (long double)a * (long double)b;
    return legacy ? (long double)(float)exact : exact;
}

static void check(double got, long double ref)
{
    assert(isfinite(got));
    assert(fabsl((long double)got - ref) < 2.0e-14L * (1.0L + fabsl(ref)));
}

static void derivative_check(int kind, int legacy, float data[4][4][3][2][9], const double *out)
{
    for (int d = 1; d <= 3; d++) {
        long double sum = 0.0L;
        for (int l = 0; l < P; l++) {
            if (kind == 0) sum += product(data[d][1][2][0][l], data[0][2][1][1][l], legacy);
            if (kind == 1) sum -= product(data[0][1][0][0][l], data[d][2][1][1][l], legacy);
            if (kind == 2) sum -= product(data[d][3][1][0][l], data[0][3][2][1][l], legacy);
            if (kind == 3) sum -= product(data[0][3][1][0][l], data[d][3][2][1][l], legacy);
        }
        check(out[d - 1], sum);
    }
}

static void derivatives(int legacy)
{
    float data[4][4][3][2][9];
    double result[3];
    for (int d = 0; d < 4; d++) for (int a = 0; a < 4; a++)
        for (int k = 0; k < 3; k++) for (int m = 0; m < 2; m++)
            for (int l = 0; l < P; l++)
                data[d][a][k][m][l] = (float)(sin(d * 0.71 + a * 0.37 + k * 0.83 + m * 0.23 + l * 0.63) * 0.7);
    /* DERIVATIVE_CALLS */
}

static void case1(int legacy, double *out)
{
    Force4B_GpuCase1Kernel(P, flat, halo, stride, 2, items,
                          ka, kb, is_halo, weights, out);
    for (int p = 0; p < 2; ++p)
        for (int d = 1; d <= 3; ++d) {
            long double sum = 0;
            for (int m = 0; m < items[p].ian; ++m)
                for (int n = 0; n < items[p].jan; ++n) {
                    long double dot = 0;
                    for (int kk = 0; kk < items[p].k_n; ++kk) {
                        int k = items[p].k_off + kk;
                        const float *a = flat + d * stride + ka[k] + m * P;
                        const float *b = (is_halo[k] ? halo : flat) + kb[k] + n * P;
                        for (int l = 0; l < P; ++l) dot += product(a[l], b[l], legacy);
                    }
                    sum += weights[items[p].cdm_off + m * items[p].jan + n] * dot;
                }
            check(out[3 * p + d - 1], sum);
        }
}

static void case2(int legacy, double *out)
{
    const size_t h[] = {0, P}, q[] = {2 * P, 4 * P};
    Force4B_GpuCase2Kernel(P, c2, stride, 2, items, h, q, weights, out);
    for (int p = 0; p < 2; ++p)
        for (int d = 1; d <= 3; ++d) {
            long double sum = 0;
            for (int m = 0; m < items[p].ian; ++m)
                for (int n = 0; n < items[p].jan; ++n) {
                    long double dot = 0;
                    for (int l = 0; l < P; ++l) {
                        size_t hi = h[p] + m * P + l, qi = q[p] + n * P + l;
                        dot -= product(c2[d * stride + hi], c2[qi], legacy);
                        dot -= product(c2[hi], c2[d * stride + qi], legacy);
                    }
                    sum += weights[items[p].cdm_off + m * items[p].jan + n] * dot;
                }
            check(out[3 * p + d - 1], sum);
        }
}

static void vna_test(int legacy, double *out)
{
    const int koff[] = {0, 2}, kn[] = {2, 1}, nc[] = {2, 1}, nh[] = {2, 2};
    const size_t offset[] = {0, 4};
    const int ke[] = {0, 3, 0}, groups[] = {1, 3, 5};
    const double ene[] = {0.1, -0.2, 0.7, -0.4, 0.8, -0.3};
    vna_kernel(2, P, 3, koff, kn, nc, nh, offset, flat, halo,
               ka, kb, is_halo, ke, ene, groups, out);
    for (int p = 0; p < 2; ++p)
        for (int m = 0; m < nc[p]; ++m)
            for (int n = 0; n < nh[p]; ++n) {
                long double sum = 0;
                for (int kk = 0; kk < kn[p]; ++kk) {
                    int k = koff[p] + kk, l = 0;
                    const float *a = flat + ka[k] + m * P;
                    const float *b = (is_halo[k] ? halo : flat) + kb[k] + n * P;
                    for (int g = 0; g < 3; ++g) {
                        long double dot = 0;
                        for (int j = 0; j < groups[g]; ++j, ++l)
                            dot += product(a[l], b[l], legacy);
                        sum += ene[ke[k] + g] * dot;
                    }
                }
                check(out[offset[p] + m * nh[p] + n], sum);
            }
}

int main(void)
{
    double results[2][3][6];
    for (int i = 0; i < 4 * P * ROWS; ++i) {
        flat[i] = (float)(sin(i * 0.37) * 0.6 + 0.05);
        c2[i] = (float)(cos(i * 0.21) * 0.4 - 0.15);
    }
    for (int i = 0; i < 2 * P; ++i) halo[i] = (float)(cos(i * 0.17) * 0.3);
    assert(openmx_fp32_enabled());
    for (int legacy = 1; legacy >= 0; --legacy) {
        openmx_set_fp32_enabled(legacy);
        case1(legacy, results[legacy][0]);
        case2(legacy, results[legacy][1]);
        vna_test(legacy, results[legacy][2]);
        derivatives(legacy);
    }
    for (int kernel = 0; kernel < 3; ++kernel) {
        int different = 0;
        for (int i = 0; i < 6; ++i)
            different |= fabs(results[0][kernel][i] - results[1][kernel][i]) > 1e-10;
        assert(different); /* Inputs must actually detect FP32 products. */
    }
    openmx_set_fp32_enabled(1);
    double repeat[6];
    case1(1, repeat);
    for (int i = 0; i < 6; ++i) assert(repeat[i] == results[1][0][i]);
    puts("PASS: projector/force kernels and local/halo force/stress derivatives, FP64 products, legacy rounding and policy reset");
    return 0;
}
"""
HARNESS = HARNESS.replace("    /* DERIVATIVE_CALLS */", "\n".join(derivative_calls))

with tempfile.TemporaryDirectory(prefix="openmx-projector-precision-") as tmp:
    source, binary = Path(tmp) / "check.c", Path(tmp) / "check"
    source.write_text(
        '#include <assert.h>\n#include <math.h>\n#include <stddef.h>\n'
        '#include <stdio.h>\n#include "openmx_precision.h"\n'
        + item + "\n" + functions + HARNESS
    )
    subprocess.run(
        shlex.split(os.environ.get("CC", "cc"))
        + ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
           "-I", str(ROOT / "source"), str(source), str(ROOT / "source/openmx_precision.c"),
           "-lm", "-o", str(binary)], check=True,
    )
    subprocess.run([str(binary)], check=True)

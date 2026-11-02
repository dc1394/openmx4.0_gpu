#!/usr/bin/env python3
"""M3 assembly: combine the M2 single-GEMM microbenchmark (gemm_micro_5080.csv)
with the production-run manifests (GEMM call counts) and the v2 aggregate
(diag/total wall times) into the lightweight Amdahl model of plan sec. 14.3:

    S_pred = 1 / ((1-f) + f/s_GEMM)          (h folded into the residual)

  f      = (sum over sites: count x t_cuBLAS_iso) / T_diag(o-leg)  [or T_total]
  s_eff  = T_gemm_cuBLAS / T_gemm_G8 for the run's shape mix
  S_meas = T_o / T_g from the campaign (5-rep means, per-system-best np)

bnc216's g leg is modelled with its real mix: 193 of the 200 back-transform
calls fell back to cuBLAS (reserve policy, 16 GB), 7 ran GEMMul8.
Writes amdahl_5080.csv and prints a human-readable table.
"""
import csv
import sys

CSVFILE = sys.argv[1] if len(sys.argv) > 1 else "gemm_micro_5080.csv"

# ---- measured campaign walls (per-system-best np; seconds): the means of the
# five retained runs of each configuration at full precision, from the
# individual outputs (work/reaggregate/production_runs.csv; a mean of five
# millisecond times is exact at four decimals).  Until 2026-11-02 these were
# the two-decimal means of s5prod_thesis_result_v2.txt, which moved the third
# decimal of a few rows; the paper's Table VI uses the full-precision means.
# system: (np, total_o, diag_o, total_g, diag_g)
WALL = {
    "bcol216": (18, 351.8004, 235.4556, 228.428, 112.3618),
    "bcol384": (16, 1436.8526, 1222.0258, 795.3268, 579.9218),
    "bnc128":  (16, 380.1916, 275.6096, 269.8342, 164.6294),
    "bnc216":  (16, 1482.6166, 1262.9824, 1230.4022, 1009.2052),
    "ccol216": (18, 129.767, 13.9784, 126.5716, 10.8934),
    "ccol512": (16, 395.4022, 110.827, 356.7494, 72.3198),
    "cnc128":  (16, 136.8702, 34.1424, 132.202, 27.9474),   # cnc128: 2026-11-01 addendum
    "cnc216":  (18, 338.6114, 132.3176, 306.9654, 101.7986),
}

# ---- GEMM sites per production run: (csv label, calls, g8_calls_in_g_leg) ----
# calls from the run manifests (gemmul8.d_calls/z_calls) and the per-solve
# call structure verified in the sources; g8_calls differs from calls only
# for bnc216_back (193 reserve-policy fallbacks -> 7 G8 + 193 cuBLAS).
SITES = {
    "bcol216": [("bcol216_fwd1", 200, 200), ("bcol216_fwd2", 200, 200), ("bcol216_back", 200, 200)],
    "bcol384": [("bcol384_fwd1", 200, 200), ("bcol384_fwd2", 200, 200), ("bcol384_back", 200, 200)],
    "bnc128":  [("bnc128_fwd1", 600, 600), ("bnc128_fwd2", 600, 600), ("bnc128_back", 200, 200)],
    "bnc216":  [("bnc216_fwd1", 600, 600), ("bnc216_fwd2", 600, 600), ("bnc216_back", 200, 7)],
    "ccol216": [("ccol216_fwd1", 25, 25), ("ccol216_fwd2", 25, 25), ("ccol216_back", 24, 24),
                ("ccol216_back1", 1, 1)],
    "ccol512": [("ccol512_fwd1", 25, 25), ("ccol512_fwd2", 25, 25), ("ccol512_back", 24, 24),
                ("ccol512_back1", 1, 1)],
    "cnc128":  [("cnc128_dfwd1", 150, 150), ("cnc128_dfwd2", 150, 150), ("cnc128_zback", 24, 24),
                ("cnc128_zback1", 1, 1)],   # manifests s5p_cnc128_g11..g15: d_calls 300, z_calls 25, 0 fallbacks
    "cnc216":  [("cnc216_dfwd1", 150, 150), ("cnc216_dfwd2", 150, 150), ("cnc216_zback", 24, 24),
                ("cnc216_zback1", 1, 1)],
}

rows = {}
with open(CSVFILE) as f:
    for r in csv.DictReader(f):
        rows[r["label"]] = r

def t_ms(label, col):
    return float(rows[label][col])

out = []
print(f"{'system':<9} {'T_gemm_cu':>9} {'T_gemm_g8':>9} {'s_eff':>6} "
      f"{'f_diag':>6} {'Sp_diag':>7} {'Sm_diag':>7} {'f_tot':>6} {'Sp_tot':>6} {'Sm_tot':>6}")
for sys_name, sites in SITES.items():
    np_, total_o, diag_o, total_g, diag_g = WALL[sys_name]
    t_cu = sum(c * t_ms(lbl, "cublas_ms_med") for lbl, c, _ in sites) / 1000.0
    # g leg: g8-run calls at g8 time, fallback remainder at cublas time
    t_g8 = sum((g8c * t_ms(lbl, "g8_ms_med") + (c - g8c) * t_ms(lbl, "cublas_ms_med")) for lbl, c, g8c in sites) / 1000.0
    s_eff = t_cu / t_g8
    f_d = t_cu / diag_o
    f_t = t_cu / total_o
    sp_d = 1.0 / ((1.0 - f_d) + f_d / s_eff)
    sp_t = 1.0 / ((1.0 - f_t) + f_t / s_eff)
    sm_d = diag_o / diag_g
    sm_t = total_o / total_g
    print(f"{sys_name:<9} {t_cu:>9.2f} {t_g8:>9.2f} {s_eff:>6.2f} "
          f"{f_d:>6.3f} {sp_d:>7.3f} {sm_d:>7.3f} {f_t:>6.3f} {sp_t:>6.3f} {sm_t:>6.3f}")
    out.append(dict(system=sys_name, np=np_, T_gemm_cublas_s=round(t_cu, 3), T_gemm_g8mix_s=round(t_g8, 3),
                    s_eff=round(s_eff, 3), f_diag=round(f_d, 4), S_pred_diag=round(sp_d, 3),
                    S_meas_diag=round(sm_d, 3), f_total=round(f_t, 4), S_pred_total=round(sp_t, 3),
                    S_meas_total=round(sm_t, 3)))

# hypothetical: bnc216 with zero fallbacks (H100-like memory headroom)
sites = SITES["bnc216"]
t_cu = sum(c * t_ms(lbl, "cublas_ms_med") for lbl, c, _ in sites) / 1000.0
t_g8_full = sum(c * t_ms(lbl, "g8_ms_med") for lbl, c, _ in sites) / 1000.0
_, total_o, diag_o, _, _ = WALL["bnc216"]
s_eff = t_cu / t_g8_full
f_d = t_cu / diag_o
sp_d = 1.0 / ((1.0 - f_d) + f_d / s_eff)
print(f"{'bnc216*':<9} {t_cu:>9.2f} {t_g8_full:>9.2f} {s_eff:>6.2f} {f_d:>6.3f} {sp_d:>7.3f} "
      f"{'---':>7}   (* zero-fallback hypothetical)")
out.append(dict(system="bnc216_nofallback", np=16, T_gemm_cublas_s=round(t_cu, 3),
                T_gemm_g8mix_s=round(t_g8_full, 3), s_eff=round(s_eff, 3), f_diag=round(f_d, 4),
                S_pred_diag=round(sp_d, 3), S_meas_diag="", f_total="", S_pred_total="", S_meas_total=""))

with open("amdahl_5080.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("\nwrote amdahl_5080.csv")

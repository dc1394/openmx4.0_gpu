#!/usr/bin/env python3
"""cnc128 row of the Amdahl-type model (same formulas as analysis.py):
   f = sum_j c_j t_j,cuBLAS / T_diag(o);  s_eff = A/B;  S_pred = 1/((1-f)+f/s_eff)
walls: means of the five retained runs at full precision (np16; work/reaggregate/production_runs.csv):
total_o 136.8702, diag_o 34.1424, total_g 132.202, diag_g 27.9474 (until 2026-11-02 the two-decimal 136.87/34.14/132.20/27.95)
counts: run manifests s5p_cnc128_g11..g15: d_calls 300 (150 NN + 150 TN), z_calls 25 (24 back + 1 SCF1 back), 0 fallbacks"""
import csv, sys
csvfile = sys.argv[1]
rows = {r["label"]: r for r in csv.DictReader(open(csvfile))}
np_, total_o, diag_o, total_g, diag_g = 16, 136.8702, 34.1424, 132.202, 27.9474
sites = [("cnc128_dfwd1", 150, 150), ("cnc128_dfwd2", 150, 150), ("cnc128_zback", 24, 24), ("cnc128_zback1", 1, 1)]
t_cu = sum(c * float(rows[l]["cublas_ms_med"]) for l, c, _ in sites) / 1000.0
t_g8 = sum(g * float(rows[l]["g8_ms_med"]) + (c - g) * float(rows[l]["cublas_ms_med"]) for l, c, g in sites) / 1000.0
s_eff = t_cu / t_g8
f_d, f_t = t_cu / diag_o, t_cu / total_o
sp_d = 1.0 / ((1.0 - f_d) + f_d / s_eff)
sp_t = 1.0 / ((1.0 - f_t) + f_t / s_eff)
sm_d, sm_t = diag_o / diag_g, total_o / total_g
delta = diag_g / diag_o - ((1.0 - f_d) + f_d / s_eff)
print(f"cnc128 np={np_} T_gemm_cu={t_cu:.3f}s T_gemm_g8={t_g8:.3f}s s_eff={s_eff:.3f} f_diag={f_d:.4f} "
      f"S_pred_diag={sp_d:.3f} S_meas_diag={sm_d:.3f} err={100*(sp_d/sm_d-1):+.2f}% f_total={f_t:.4f} "
      f"S_pred_total={sp_t:.3f} S_meas_total={sm_t:.3f} delta={delta:.4f}")
print("amdahl csv row:", ",".join(str(v) for v in ["cnc128", np_, round(t_cu, 3), round(t_g8, 3), round(s_eff, 3), round(f_d, 4),
      round(sp_d, 3), round(sm_d, 3), round(f_t, 4), round(sp_t, 3), round(sm_t, 3)]))
print(f"table row: cnc128 & {f_d:.3f} & {s_eff:.2f} & {sp_d:.3f} & {sm_d:.3f} & {100*(sp_d/sm_d-1):+.2f}\\\\")

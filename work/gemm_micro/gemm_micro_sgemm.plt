# Fig.-6 source: single-GEMM speedup s_GEMM = t_cuBLAS-FP64 / t_GEMMul8
# (median of 7 interleaved reps, square NN, isolated GPU) vs matrix size,
# RTX 5080 (sm_120) and H100 PCIe (sm_90), D = real / Z = complex.
# Data: gemm_micro_sgemm.pdat from gemm_micro_{5080,h100}.csv.
# House style: pngcairo noenhanced 1500x900, #eb6834 5080 / #2a78d6 H100.
set terminal pngcairo noenhanced size 1500,900 font ",13"
set output "gemm_micro_sgemm.png"
set datafile missing "?"

set title "GEMMul8 (INT8, 15 moduli) vs cuBLAS FP64, single GEMM: s_GEMM by matrix size and GPU"
set xlabel "matrix dimension n (square, alpha=1 beta=0)"
set ylabel "s_GEMM = t_cuBLAS / t_GEMMul8   (>1 : GEMMul8 faster)"
set log y
set yrange [0.15:30]
set xrange [800:8200]
set grid ytics lc rgb "#dddddd"
set key top left

set arrow from 800,1 to 8200,1 nohead dt 3 lc rgb "#888888" lw 2

plot "gemm_micro_sgemm.pdat" using 1:2 with linespoints      lc rgb "#eb6834" lw 2 pt 7 ps 1.4 title "RTX 5080, DGEMM", \
     ""                      using 1:3 with linespoints dt 2 lc rgb "#eb6834" lw 2 pt 5 ps 1.4 title "RTX 5080, ZGEMM", \
     ""                      using 1:4 with linespoints      lc rgb "#2a78d6" lw 2 pt 7 ps 1.4 title "H100 PCIe, DGEMM", \
     ""                      using 1:5 with linespoints dt 2 lc rgb "#2a78d6" lw 2 pt 5 ps 1.4 title "H100 PCIe, ZGEMM"

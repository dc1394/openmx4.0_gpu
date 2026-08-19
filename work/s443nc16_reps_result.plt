# Stacked-bar view of work/s443nc16_reps_result.txt (means over 3 reps).
# Bar height = Total Computational Time; bottom segment = Diagonalization,
# top segment = the rest (Total - Diagonalization).  Whiskers = +-1 sample SD
# of Total.  Groups: 1-4 nodes; within a group: gpusolver + MPS with gemmul8
# on, CPU elpa2, and gpusolver2 (distributed ELPA 2026.02 + COSMA).
#
# 16 ranks/node EVERYWHERE -- no annotated exception, unlike the 36-rank
# chart whose 1-node gpusolver2 bar had to be taken at 16.  One binary
# throughout as well (gpusolver2 merge, f5fb5b3), so no bar in this figure
# is compared across a binary boundary.
#
# The 2-node gpusolver2 whisker is wide because s443nc_2n16_d3 lost ~25 s to
# shared-filesystem contention outside the solver (readfile, Mulliken_Charge,
# RestartFileDFT); its Diagonalization SD is 0.65 s.  Kept as measured.
#
# Render:  gnuplot s443nc16_reps_result.plt   (writes s443nc16_reps_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "s443nc16_reps_result.png"

# columns: x  total  diag  sd_total
$data << EOD
 0   184.69    94.70   1.06
 1  1750.84  1342.48   2.46
 2   667.14   566.45   6.02
 4   147.64    90.08   0.14
 5   963.40   749.96   2.25
 6   421.21   358.79  14.23
 8   134.75    89.05   0.88
 9   702.43   552.81   3.82
10   314.91   272.08   0.79
12   127.81    88.03   0.15
13   560.81   447.18   1.74
14   262.00   225.40   1.64
EOD

set title "sidia443_nc_cluster at 16 ranks/node: Total = Diagonalization + rest" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "384 Si non-collinear, cluster solver (Γ-only, matrix dim 9984), 16 ranks/node on whole 48-core nodes; bars = mean of 3 reps, whiskers = ±1 SD of Total" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "gpusolver = single-GPU solver + MPS, scf.gemmul8.enable on;  elpa2 = CPU only;  gpusolver2 = distributed ELPA 2026.02 + COSMA;  one binary throughout (f5fb5b3)" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:1900]
set xrange [-0.8:14.8]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",11" textcolor rgb "#52514e" \
    ("gpusolver" 0, "elpa2" 1, "gpusolver2" 2, \
     "gpusolver" 4, "elpa2" 5, "gpusolver2" 6, \
     "gpusolver" 8, "elpa2" 9, "gpusolver2" 10, \
     "gpusolver" 12, "elpa2" 13, "gpusolver2" 14)

set label 1 "1 node"  at first  1, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"
set label 2 "2 nodes" at first  5, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"
set label 3 "3 nodes" at first  9, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"
set label 4 "4 nodes" at first 13, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"

# the one wide whisker, so it is not read as solver variability
set label 5 "The wide 2-node gpusolver2 whisker is one rep that lost ~25 s to shared-filesystem contention outside the solver (readfile, Mulliken_Charge, RestartFileDFT); its Diagonalization SD is 0.65 s." \
    at graph 0.5, graph -0.155 center front font ",11" textcolor rgb "#898781"

set bmargin 8
set tmargin 6

set key top right samplen 1.6 spacing 1.5 font ",13" textcolor rgb "#52514e"

set boxwidth 0.85
set style fill solid 1.0 border rgb "#fcfcfb"

plot $data using 1:2 with boxes linecolor rgb "#eb6834" linewidth 2 \
         title "Other (Total - Diagonalization)", \
     $data using 1:3 with boxes linecolor rgb "#2a78d6" linewidth 2 \
         title "Diagonalization", \
     $data using 1:2:4 with yerrorbars linecolor rgb "#52514e" linewidth 1.5 \
         pointtype 7 pointsize 0 notitle, \
     $data using 1:($2+$4+45):(sprintf("%.1f",$2)) with labels \
         font ",11" textcolor rgb "#52514e" notitle

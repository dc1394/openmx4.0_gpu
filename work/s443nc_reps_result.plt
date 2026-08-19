# Stacked-bar view of work/s443nc_reps_result.txt (means over 3 reps).
# Bar height = Total Computational Time; bottom segment = Diagonalization,
# top segment = the rest (Total - Diagonalization).  Whiskers = +-1 sample SD
# of Total.  Groups: 1-4 nodes; within a group: GPU gpusolver + gemmul8-on
# ("on"), the same with gemmul8-off ("off"), CPU elpa2 ("CPU"), and
# gpusolver2 = distributed ELPA+COSMA ("g2").  36 ranks/node throughout,
# EXCEPT the 1-node g2 bar: gpusolver2 is host-OOM-killed at 36, 24 and 48
# ranks/node on these 124-GiB nodes, so that one point is 16 ranks/node and
# is marked "16rk" -- it is the largest 1-node configuration that runs, not
# a rank-matched comparison.
# The 1-node "off" point is the 2026-08-14 re-measurement (o4/o5/o6, clean
# nodes); the original triple contained a 217 s slow-node rep (bnode033) and
# is preserved in s443nc_reps_result.txt.bak.  The g2 columns were added
# 2026-08-15 (binary f5fb5b3); the 4-node g2 triple is d2/d3/d8, the first
# three reps whose full node set avoided bnode013/bnode033.
#
# Render:  gnuplot s443nc_reps_result.plt   (writes s443nc_reps_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "s443nc_reps_result.png"

# columns: x  total  diag  sd_total
$data << EOD
 0   183.79   93.63  1.60
 1   185.19   95.61  0.48
 2  1060.59  842.71  2.09
 3   667.14  566.45  6.02
 5   154.88   93.75  1.53
 6   154.69   93.27  3.97
 7   671.79  537.88  6.30
 8   381.56  325.11  1.91
10   139.53   92.44  0.37
11   139.94   92.57  1.41
12   501.38  407.31  1.82
13   321.26  274.15  2.93
15   134.30   91.21  0.75
16   135.40   90.87  3.24
17   425.17  349.90  0.79
18   281.34  239.66  1.33
EOD

set title "sidia443_nc_cluster node scaling: Total = Diagonalization + rest" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "384 Si non-collinear, cluster solver (Γ-only, matrix dim 9984), 36 ranks/node; bars = mean of 3 reps, whiskers = ±1 SD of Total" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "on / off = gpusolver + MPS with scf.gemmul8.enable on / off;  CPU = elpa2;  g2 = gpusolver2 (distributed ELPA 2026.02 + COSMA);  binaries e227507 / f5fb5b3" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:1150]
set xrange [-0.9:18.9]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",12" textcolor rgb "#52514e" \
    ("on" 0, "off" 1, "CPU" 2, "g2" 3, "on" 5, "off" 6, "CPU" 7, "g2" 8, \
     "on" 10, "off" 11, "CPU" 12, "g2" 13, "on" 15, "off" 16, "CPU" 17, "g2" 18)

set label 1 "1 node"  at first  1.5, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
set label 2 "2 nodes" at first  6.5, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
set label 3 "3 nodes" at first 11.5, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
set label 4 "4 nodes" at first 16.5, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"

# the one bar that is not 36 ranks/node
set label 5 "16rk" at first 3, 667.14+80 center font ",10" textcolor rgb "#898781"

set bmargin 6
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
     $data using 1:($2+$4+30):(sprintf("%.1f",$2)) with labels \
         font ",11" textcolor rgb "#52514e" notitle

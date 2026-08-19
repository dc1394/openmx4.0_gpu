# Stacked-bar view of work/dia64dc_nodes_result.txt (means over 3 reps).
# Bar height = Total Computational Time; bottom segment = Diagonalization,
# top segment = the rest (Total - Diagonalization).  Whiskers = +-1 SD of
# Total.  Groups: 1-4 nodes; within a group: gpusolver + MPS with gemmul8 on,
# the same with gemmul8 off, and CPU elpa2.  48 ranks/node throughout.
#
# dia64_dc-lno: 64 C atoms, 832 basis functions, scf.EigenvalueSolver dc-lno,
# MD.Type Opt with MD.maxIter 1.  One binary throughout (f5fb5b3).
#
# Render:  gnuplot dia64dc_nodes_result.plt   (writes dia64dc_nodes_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "dia64dc_nodes_result.png"

# columns: x  total  diag  sd_total
$data << EOD
 0  203.36  176.62  1.04
 1  200.14  173.01  1.43
 2  612.70  570.27  1.49
 4  116.01   94.08  0.97
 5  113.57   92.35  2.59
 6  305.92  275.21  1.63
 8   85.12   65.45  0.38
 9   84.41   64.53  0.66
10  242.48  211.82  1.61
12   68.94   49.71  0.42
13   67.62   48.83  0.66
14  211.96  184.15  1.51
EOD

set title "dia64_dc-lno node scaling: Total = Diagonalization + rest" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "64 C atoms, 832 basis functions, DC-LNO solver (Γ-only), 48 ranks/node; bars = mean of 3 reps, whiskers = ±1 SD of Total" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "on / off = gpusolver + MPS with scf.gemmul8.enable on / off;  CPU = elpa2;  one binary throughout (f5fb5b3)" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:680]
set xrange [-0.8:14.8]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",11" textcolor rgb "#52514e" \
    ("g8 on" 0, "g8 off" 1, "CPU" 2,  "g8 on" 4, "g8 off" 5, "CPU" 6, \
     "g8 on" 8, "g8 off" 9, "CPU" 10, "g8 on" 12, "g8 off" 13, "CPU" 14)

set label 1 "1 node"  at first  1, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"
set label 2 "2 nodes" at first  5, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"
set label 3 "3 nodes" at first  9, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"
set label 4 "4 nodes" at first 13, graph -0.09 center font ",14" textcolor rgb "#0b0b0b"

set label 5 "DC-LNO parallelises over the 64 atoms, so atom-owning ranks cap at min(48n, 64): 48 at 1 node, 64 at 2, 3 and 4 nodes.\nGains past 2 nodes come from spreading those 64 owners thinner across more nodes, not from putting more ranks on the work." \
    at graph 0.5, graph -0.175 center front font ",11" textcolor rgb "#898781"

set bmargin 9
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
     $data using 1:($2+$4+16):(sprintf("%.1f",$2)) with labels \
         font ",11" textcolor rgb "#52514e" notitle

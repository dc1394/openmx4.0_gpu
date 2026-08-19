# Stacked-bar view of work/s333nc_reps_result.txt (means over 3 reps).
# Bar height = Total Computational Time; bottom segment = Diagonalization,
# top segment = the rest (Total - Diagonalization).  Whiskers = +-1 sample SD
# of Total.  Groups: 1-4 nodes; within a group: GPU gemmul8-on ("on"),
# GPU gemmul8-off ("off"), CPU elpa2 ("CPU").  48 ranks/node throughout.
# 1-node "on" is the 2026-08-14 re-measurement (g4/g5/g6, clean nodes).
#
# Render:  gnuplot s333nc_reps_result.plt   (writes s333nc_reps_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "s333nc_reps_result.png"

# columns: x  total  diag  sd_total
$data << EOD
 0  246.77 158.86 0.55
 1  242.31 154.55 0.18
 2  928.22 812.56 7.23
 4  158.06  94.19 1.48
 5  158.07  94.21 3.48
 6  542.22 459.65 2.06
 8  137.13  76.59 1.56
 9  139.31  77.95 2.78
10  399.04 334.04 3.81
12  123.52  63.02 0.68
13  121.91  62.43 1.54
14  328.22 265.69 2.26
EOD

set title "sidia333_nc node scaling: Total = Diagonalization + rest" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "216 Si non-collinear, band solver (k-grid 2x2x2), 48 ranks/node; bars = mean of 3 reps, whiskers = ±1 SD of Total" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "on / off = GPU gpusolver + MPS with scf.gemmul8.enable on / off;  CPU = scf.eigen.lib elpa2;  binary e227507" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:1000]
set xrange [-0.9:14.9]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",13" textcolor rgb "#52514e" \
    ("on" 0, "off" 1, "CPU" 2, "on" 4, "off" 5, "CPU" 6, \
     "on" 8, "off" 9, "CPU" 10, "on" 12, "off" 13, "CPU" 14)

set label 1 "1 node"  at first  1, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
set label 2 "2 nodes" at first  5, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
set label 3 "3 nodes" at first  9, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
set label 4 "4 nodes" at first 13, graph -0.085 center font ",14" textcolor rgb "#0b0b0b"
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
     $data using 1:($2+$4+26):(sprintf("%.1f",$2)) with labels \
         font ",11" textcolor rgb "#52514e" notitle

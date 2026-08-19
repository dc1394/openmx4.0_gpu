# Stacked-bar view of work/siband_scaling_result.txt (means over clean reps).
# Bar height = Total (25 fixed SCF); bottom segment = Diagonalization; top =
# rest.  Whiskers = +-1 SD of Total.  Groups: col300 and nc216 at 1/2/4
# nodes; within a group: cuBLAS (gemmul8 off) vs GEMMul8 (on).
#
# Si diamond BAND solver, 48 ranks/node, -nt 1, MPS on, 8 computed k points
# (8/4/2 k per GPU at 1/2/4 nodes).  Binary cd5f0d5 + GEMMul8 v3.2.0.
#
# Render:  gnuplot siband_scaling_result.plt  (writes siband_scaling_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "siband_scaling_result.png"

# columns: x  total  diag  sd_total
$data << EOD
 0  130.93   82.24  1.31
 1  147.64   79.73  0.41
 3   87.84   47.56  1.35
 4   85.37   45.24  0.16
 6   68.73   31.11  1.41
 7   68.07   29.81  1.88
10  252.50  154.01  0.79
11  259.36  158.99  2.87
13  167.67   94.18  1.32
14  166.03   92.65  1.28
16  139.75   64.55  1.41
17  139.62   64.41  1.14
EOD

set title "Si diamond band solvers: limited strong scaling, Total = Diagonalization + rest" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "col = 300 atoms (n=3900, spin off) | NC = 216 atoms (2n=5616);  25 fixed SCF, 48 ranks/node, MPS on, 1 H100/node;  bars = mean, whiskers = ±1 SD of Total" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "cuB = scf.gemmul8.enable off (cuBLAS FP64);  G8 = gemmul8 on (INT8 emulation);  binary cd5f0d5 + GEMMul8 v3.2.0 throughout" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:300]
set xrange [-0.9:17.9]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",11" textcolor rgb "#52514e" \
    ("cuB" 0, "G8" 1, "cuB" 3, "G8" 4, "cuB" 6, "G8" 7, \
     "cuB" 10, "G8" 11, "cuB" 13, "G8" 14, "cuB" 16, "G8" 17)

set label 1 "1 node"  at first  0.5, graph -0.09 center font ",13" textcolor rgb "#0b0b0b"
set label 2 "2 nodes" at first  3.5, graph -0.09 center font ",13" textcolor rgb "#0b0b0b"
set label 3 "4 nodes" at first  6.5, graph -0.09 center font ",13" textcolor rgb "#0b0b0b"
set label 4 "1 node"  at first 10.5, graph -0.09 center font ",13" textcolor rgb "#0b0b0b"
set label 5 "2 nodes" at first 13.5, graph -0.09 center font ",13" textcolor rgb "#0b0b0b"
set label 6 "4 nodes" at first 16.5, graph -0.09 center font ",13" textcolor rgb "#0b0b0b"

set label 7 "collinear band, 300 Si atoms" at first 3.5, graph -0.135 center font ",13" textcolor rgb "#52514e"
set label 8 "non-collinear band, 216 Si atoms" at first 13.5, graph -0.135 center font ",13" textcolor rgb "#52514e"

set label 9 "8 computed k points parallelise as 8/4/2 k per GPU at 1/2/4 nodes; the Diagonalization segment scales, the rest does not (Amdahl).\nGEMMul8 vs cuBLAS: +13% Total at col 1 node (Set_Hamiltonian slowdown, see result txt), within noise everywhere else; Utot differs by <= 4e-12 Ha." \
    at graph 0.5, graph -0.185 center front font ",11" textcolor rgb "#898781"

set bmargin 9.5
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
     $data using 1:($2+$4+9):(sprintf("%.1f",$2)) with labels \
         font ",11" textcolor rgb "#52514e" notitle

# Stacked-bar view of work/siprod_result.txt: the H100 1-node production
# matrix (Layer 1 CPU-vs-GPU, Layer 3 cuBLAS-vs-GEMMul8, MPS ablation).
# Bar = Total (25 fixed SCF, mean over 3-5 clean reps); bottom segment =
# Diagonalization; whiskers = +-1 SD of Total.
# CPU = elpa2 flat 48 ranks; cuB/G8 = gpusolver + MPS with gemmul8 off/on;
# noMPS = gpusolver, MPS off, gemmul8 off (plan 8.3 ablation, NC only).
#
# Render:  gnuplot siprod_result.plt   (writes siprod_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "siprod_result.png"

# columns: x total diag sd
$data << EOD
 0  239.28  169.74  1.18
 1   81.49   37.00  2.11
 2   81.12   36.31  2.21
 4  489.09  408.50  0.41
 5  131.22   82.62  0.40
 6  147.67   80.11  0.89
 8  945.60  815.11  3.78
 9  253.39  154.53  1.74
10  255.35  157.04  1.03
11  424.03  196.83  1.81
13   87.55   17.21  1.48
14   51.02    7.35  0.48
15   51.52    8.01  0.81
17  139.75   47.04  0.02
18   69.43   15.80  0.56
19   70.35   16.27  0.36
21  324.50  193.63  3.78
22   99.21   30.59  0.83
23   99.25   30.66  0.96
24  227.00   30.32  0.82
26  564.81  419.41  1.61
27  136.01   57.19  1.30
28  135.48   56.72  0.48
EOD

set title "Si diamond, H100 1-node matrix: CPU vs GPU, GEMMul8, and MPS across the four dense solvers" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "25 fixed SCF, 48 ranks (-nt 1), 1 node; bars = mean of 3-5 clean reps, whiskers = ±1 SD of Total;  CPU = elpa2 flat MPI (best config)" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "cuB / G8 = gpusolver + MPS, scf.gemmul8.enable off / on;  noMPS = cuBLAS without MPS (ablation);  binary cd5f0d5 + GEMMul8 v3.2.0" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:1000]
set xrange [-0.9:28.9]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",10" textcolor rgb "#52514e" rotate by 0 \
    ("CPU" 0, "cuB" 1, "G8" 2, "CPU" 4, "cuB" 5, "G8" 6, \
     "CPU" 8, "cuB" 9, "G8" 10, "noMPS" 11, \
     "CPU" 13, "cuB" 14, "G8" 15, "CPU" 17, "cuB" 18, "G8" 19, \
     "CPU" 21, "cuB" 22, "G8" 23, "noMPS" 24, "CPU" 26, "cuB" 27, "G8" 28)

set label 1 "band col\n216 (n=2808)"     at first  1,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 2 "band col\n300 (n=3900)"     at first  5,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 3 "band NC\n216 (2n=5616)"     at first  9.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 4 "cluster col\n216"           at first 14,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 5 "cluster col\n360 (n=4680)"  at first 18,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 6 "cluster NC\n216"            at first 22.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 7 "cluster NC\n300 (2n=7800)"  at first 27,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"

set label 9 "Layer 1 (CPU/cuB Total): 2.94 / 3.73 / 3.73 / 1.72 / 2.01 / 3.27 / 4.15x.   Layer 3: GEMMul8 significant only at band col 300 (1.125x slower Total, Set_Hamiltonian interference).\nMPS off/on: 1.67x (band NC) and 2.29x (cluster NC) slower without MPS -- Diagonalization unchanged on cluster; the loss sits in the 48-rank shared phases (Set_Ham, grids)." \
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
     $data using 1:($2+$4+28):(sprintf("%.0f",$2)) with labels \
         font ",10" textcolor rgb "#52514e" notitle

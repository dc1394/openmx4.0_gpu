# Stacked-bar view of work/siprod_thesis_result.txt: THESIS (v2.0_thesis tag)
# matrix (Layer 1 CPU-vs-GPU, Layer 3 cuBLAS-vs-GEMMul8, MPS ablation).
# Bar = Total (25 fixed SCF, mean over 3-5 clean reps); bottom segment =
# Diagonalization; whiskers = +-1 SD of Total.
# CPU = elpa2 flat 48 ranks; cuB/G8 = gpusolver + MPS with gemmul8 off/on;
# noMPS = gpusolver, MPS off, gemmul8 off (plan 8.3 ablation, NC only).
#
# Render:  gnuplot siprod_result.plt   (writes siprod_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "siprod_thesis_result.png"

# columns: x total diag sd
$data << EOD
 0   240.48   170.16   0.75
 1    81.45    37.08   0.87
 2    81.33    36.77   1.77
 4   489.47   408.92   1.53
 5   131.37    83.04   1.76
 6   151.03    81.65   6.39
 8   945.85   817.50   2.70
 9   252.14   153.81   1.46
10   255.82   157.21   1.64
11   423.21   196.06   2.76
13    87.45    17.45   0.55
14    51.35     8.29   1.28
15    51.90     8.14   0.73
17   139.97    46.78   0.54
18    69.56    16.01   0.83
19    69.53    16.22   0.31
21   321.99   193.68   3.77
22   100.52    31.10   3.11
23    99.91    30.85   0.77
24   228.82    31.41   0.66
26   563.80   420.67   6.60
27   135.52    57.03   0.71
28   135.13    56.88   1.52
EOD

set title "Si diamond, H100 1-node THESIS matrix (v2.0_thesis): CPU vs GPU, GEMMul8, MPS -- four dense solvers" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "25 fixed SCF, 48 ranks (-nt 1), 1 node; bars = mean of 3-5 clean reps, whiskers = ±1 SD of Total;  CPU = elpa2 flat MPI (best config)" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "cuB / G8 = gpusolver + MPS, scf.gemmul8.enable off / on;  noMPS = cuBLAS without MPS (ablation);  binary v2.0_thesis (md5 11227640), GEMMul8 833e576" \
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

set label 9 "Layer 1 (CPU/cuB Total): 2.95 / 3.73 / 3.75 / 1.70 / 2.01 / 3.20 / 4.16x.   Layer 3: bcol300 1.150x slower Total (Set_Hamiltonian interference); bnc216 1.015x (small but significant); all others n.s.\nMPS off/on: 1.68x (band NC) and 2.28x (cluster NC) slower without MPS -- cluster Diag unchanged; every run manifest-verified: v2.0_thesis, GPU dense path, zero fallbacks." \
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

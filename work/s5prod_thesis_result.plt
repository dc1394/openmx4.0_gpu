# Stacked-bar view of work/s5prod_thesis_result.txt: RTX 5080 THESIS matrix
# (Layer 3 cuBLAS-vs-GEMMul8, MPS ablation, CPU np16 reference).
# Bar = Total (25 fixed SCF, mean over 3-5 clean reps); bottom segment =
# Diagonalization; whiskers = +-1 SD of Total.
# CPU = elpa2 flat 16 ranks (reference); cuB/G8 = gpusolver + MPS with
# gemmul8 off/on; noMPS = same without MPS (plan 8.3 ablation).
#
# Render:  gnuplot s5prod_thesis_result.plt   (writes s5prod_thesis_result.png)

set terminal pngcairo noenhanced size 1500,900 font "DejaVu Sans,14" background rgb "#fcfcfb"
set output "s5prod_thesis_result.png"

# columns: x total diag sd
$data << EOD
 0    86.09    37.20   0.40
 1    89.21    32.82   0.30
 3   380.19   275.61   1.03
 4   269.83   164.63   1.41
 6  4839.45  4534.67   1.73
 7  1482.62  1262.98   3.77
 8  1230.40  1009.21   4.75
 9  1502.71  1273.89   2.21
11    58.55     7.30   0.20
12    63.50     7.50   0.43
14   136.87    34.14   0.70
15   132.20    27.95   2.10
17   804.86   501.44   0.76
18   347.98   132.54   2.19
19   318.16   101.92   1.82
20   355.71   132.56   2.79
21   325.91   101.88   3.35
23   622.34   479.99   0.88
24   356.24   235.61   0.40
25   233.34   112.74   0.22
27  1436.85  1222.03   2.25
28   795.33   579.92   1.08
30   178.07    36.07   0.42
31   134.23    13.80   0.42
32   131.34    10.76   0.72
34   395.40   110.83   0.50
35   356.75    72.32   0.81
EOD

set title "Si diamond, RTX 5080 THESIS matrix (v2.0_thesis): GEMMul8, MPS, CPU np16 reference -- four dense solvers" \
    font ",18" textcolor rgb "#0b0b0b" offset 0,2.2
set label 100 "25 fixed SCF, 16 ranks (-nt 1), Core i9-10980XE + RTX 5080 16 GB; bars = mean of 3-5 clean reps, whiskers = ±1 SD of Total;  CPU = elpa2 flat MPI (np16 reference)" \
    at graph 0.5, graph 1.075 center font ",12" textcolor rgb "#52514e"
set label 101 "cuB / G8 = gpusolver + MPS, scf.gemmul8.enable off / on;  noMPS = without MPS (ablation);  binary v2.0_thesis (md5 e3e64c35), GEMMul8 v3.2.0 (833e576)" \
    at graph 0.5, graph 1.033 center font ",12" textcolor rgb "#52514e"

set ylabel "wall-clock seconds (Max_Time column, slowest rank)" \
    font ",13" textcolor rgb "#52514e" offset 1,0
set yrange [0:1600]
set xrange [-0.9:35.9]

set border 3 linecolor rgb "#c3c2b7" linewidth 1
set grid ytics back linecolor rgb "#e1e0d9" linewidth 1
set ytics nomirror out scale 0.5 font ",12" textcolor rgb "#898781"
set xtics nomirror scale 0 font ",10" textcolor rgb "#52514e" rotate by 0 \
    ("cuB" 0, "G8" 1, "cuB" 3, "G8" 4, \
     "CPU" 6, "cuB" 7, "G8" 8, "noMPS" 9, \
     "cuB" 11, "G8" 12, "cuB" 14, "G8" 15, \
     "CPU" 17, "cuB" 18, "G8" 19, "noMPS" 20, "G8\nnoMPS" 21, \
     "CPU" 23, "cuB" 24, "G8" 25, "cuB" 27, "G8" 28, \
     "CPU" 30, "cuB" 31, "G8" 32, "cuB" 34, "G8" 35)

set label 1 "band NC 64\n(2n=1664)"    at first  0.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 2 "band NC 128\n(2n=3328)"   at first  3.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 3 "band NC 216\n(2n=5616)"   at first  7.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 4 "cluster NC\n64"           at first 11.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 5 "cluster NC\n128"          at first 14.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 6 "cluster NC\n216"          at first 19,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 7 "band col 216\n(n=2808)"   at first 24,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 8 "band col 384\n(n=4992)"   at first 27.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 9 "cluster col\n216"         at first 31,   graph -0.105 center font ",12" textcolor rgb "#0b0b0b"
set label 10 "cluster col 512\n(n=6656)" at first 34.5, graph -0.105 center font ",12" textcolor rgb "#0b0b0b"

# the band-NC-216 CPU bar (4839 s) is clipped by the y-range; annotate it
set label 11 "4839 s (off scale)" at first 6, first 1420 center rotate by 90 front font ",10" textcolor rgb "#fcfcfb"

set label 12 "Layer 3 (G8/cuB Total): band NC 1.04 / 0.71 / 0.83x -- band col 0.65 / 0.55x -- cluster NC 1.09 / 0.97 / 0.91x -- cluster col 0.98 / 0.90x (64-atom points: overhead-dominated).\nMPS off/on 1.01-1.02x only (np16, 1 consumer GPU; H100-48rank saw 1.68-2.28x).  bnc216 G8: 193 reserve-policy fallbacks/run (16 GB limit, formal result); all else manifest-clean." \
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
     $data using 1:($2 > 1600 ? 1/0 : $2+$4+45):(sprintf("%.0f",$2)) with labels \
         font ",10" textcolor rgb "#52514e" notitle

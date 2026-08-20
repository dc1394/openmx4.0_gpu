# Si diamond BAND limited strong scaling (plan v2.6 sec. 7.4/8.5 + 8-node
# extension), v2.0_thesis tag build.  Data: siband_scaling_thesis_result.pdat
# (means +- 1 sample SD over the kept reps, Max_Time).  Whiskers = +-1 SD.
# House style: pngcairo noenhanced 1500x900, #2a78d6 cuBLAS / #eb6834 GEMMul8.
set terminal pngcairo noenhanced size 1500,900 font ",13"
set output "siband_scaling_thesis_result.png"

set multiplot layout 1,2 title "Si diamond band, strong scaling 1-8 nodes (48 ranks/node, 1 H100/node, MPS on; fixed 25 SCF; v2.0_thesis)"

set log x 2
set xtics (1,2,4,8)
set xrange [0.8:10]
set xlabel "nodes (1 GPU each; 8 computed k points -> 8/4/2/1 k-owners per GPU)"
set ylabel "wall time (s), Max_Time"
set key top right
set grid ytics lc rgb "#dddddd"

set title "col300: 300 atoms, collinear, n=3900"
set yrange [0:160]
plot "siband_scaling_thesis_result.pdat" index 0 using 2:11 with lines dt 3 lc rgb "#888888" lw 2 title "ideal T1/p (cuBLAS Total)", \
     "" index 0 using 2:3:4  with yerrorlines lc rgb "#2a78d6" lw 2 pt 7 ps 1.4 title "Total cuBLAS FP64", \
     "" index 0 using 2:7:8  with yerrorlines lc rgb "#eb6834" lw 2 pt 5 ps 1.4 title "Total GEMMul8", \
     "" index 0 using 2:5:6  with yerrorlines dt 2 lc rgb "#2a78d6" lw 2 pt 6 ps 1.4 title "Diag cuBLAS FP64", \
     "" index 0 using 2:9:10 with yerrorlines dt 2 lc rgb "#eb6834" lw 2 pt 4 ps 1.4 title "Diag GEMMul8"

set title "nc216: 216 atoms, non-collinear, 2n=5616"
set yrange [0:280]
plot "siband_scaling_thesis_result.pdat" index 1 using 2:11 with lines dt 3 lc rgb "#888888" lw 2 title "ideal T1/p (cuBLAS Total)", \
     "" index 1 using 2:3:4  with yerrorlines lc rgb "#2a78d6" lw 2 pt 7 ps 1.4 title "Total cuBLAS FP64", \
     "" index 1 using 2:7:8  with yerrorlines lc rgb "#eb6834" lw 2 pt 5 ps 1.4 title "Total GEMMul8", \
     "" index 1 using 2:5:6  with yerrorlines dt 2 lc rgb "#2a78d6" lw 2 pt 6 ps 1.4 title "Diag cuBLAS FP64", \
     "" index 1 using 2:9:10 with yerrorlines dt 2 lc rgb "#eb6834" lw 2 pt 4 ps 1.4 title "Diag GEMMul8"

unset multiplot

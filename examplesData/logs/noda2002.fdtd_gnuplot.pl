set term x11 persist           1
plot 'noda2002.fdtd_Wire probe 1_Wz_10_10_2_s1.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'noda2002.fdtd_Wire probe 2_Wz_10_10_4_s4.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'noda2002.fdtd_Wire probe 3_Wz_10_10_4_s4.dat' using 1:2 every 1::2 with lines
set term x11 persist           4
plot 'noda2002.fdtd_Wire probe 4_Wz_10_10_4_s4.dat' using 1:2 every 1::2 with lines
set term x11 persist           5
plot 'noda2002.fdtd_Bulk probe_Jz_9_9_2__10_10_2.dat' using 1:2 every 1::2 with lines

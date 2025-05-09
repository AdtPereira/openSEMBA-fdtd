set term x11 persist           1
plot 'noda2002.fdtd_Wire_probe_A_Wz_12_12_2_s1.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'noda2002.fdtd_Wire_probe_B_Wz_12_12_3_s2.dat' using 1:2 every 1::2 with lines

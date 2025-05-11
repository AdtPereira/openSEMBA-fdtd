set term x11 persist           1
plot 'noda2002.fdtd_Wire_probe_A_Wz_24_24_4_s1.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'noda2002.fdtd_Bulk_probe_Jz_23_23_4__24_24_4.dat' using 1:2 every 1::2 with lines

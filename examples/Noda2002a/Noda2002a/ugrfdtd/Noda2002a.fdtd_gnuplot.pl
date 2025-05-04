set term x11 persist           1
plot 'Noda2002a.fdtd_Point probe_Ex_12_12_2.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'Noda2002a.fdtd_Point probe_Ey_12_12_2.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'Noda2002a.fdtd_Point probe_Ez_12_12_2.dat' using 1:2 every 1::2 with lines
set term x11 persist           4
plot 'Noda2002a.fdtd_Wire probe_Wz_12_12_2_s1.dat' using 1:2 every 1::2 with lines
set term x11 persist           5
plot 'Noda2002a.fdtd_Bulk probe_Jz_11_11_2__12_12_2.dat' using 1:2 every 1::2 with lines

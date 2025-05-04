set term x11 persist           1
plot '2d_case.fdtd_Point probe_1_Ex_50_0_80.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot '2d_case.fdtd_Point probe_1_Ey_50_0_80.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot '2d_case.fdtd_Point probe_1_Ez_50_0_80.dat' using 1:2 every 1::2 with lines

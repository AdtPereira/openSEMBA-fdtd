set term x11 persist           1
plot 'SphereGround.fdtd_Point probe_1_Ex_50_50_50.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'SphereGround.fdtd_Point probe_1_Ey_50_50_50.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'SphereGround.fdtd_Point probe_1_Ez_50_50_50.dat' using 1:2 every 1::2 with lines

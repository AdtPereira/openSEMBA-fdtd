set term x11 persist           1
plot 'SphereRCS.fdtd_Point probe_Ex_50_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'SphereRCS.fdtd_Point probe_Ey_50_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'SphereRCS.fdtd_Point probe_Ez_50_50_13.dat' using 1:2 every 1::2 with lines

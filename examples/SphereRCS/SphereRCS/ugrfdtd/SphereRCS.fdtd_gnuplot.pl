set term x11 persist           1
plot 'SphereRCS.fdtd_Point probe_Ex_15_15_4.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'SphereRCS.fdtd_Point probe_Ey_15_15_4.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'SphereRCS.fdtd_Point probe_Ez_15_15_4.dat' using 1:2 every 1::2 with lines

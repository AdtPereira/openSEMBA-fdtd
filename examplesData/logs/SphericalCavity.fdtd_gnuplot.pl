set term x11 persist           1
plot 'SphericalCavity.fdtd_Point probe_Ex_50_25_50.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'SphericalCavity.fdtd_Point probe_Ey_50_25_50.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'SphericalCavity.fdtd_Point probe_Ez_50_25_50.dat' using 1:2 every 1::2 with lines

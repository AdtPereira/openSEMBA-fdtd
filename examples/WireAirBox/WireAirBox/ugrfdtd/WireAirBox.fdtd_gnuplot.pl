set term x11 persist           1
plot 'WireAirBox.fdtd_Point probe A_Ex_50_50_25.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'WireAirBox.fdtd_Point probe A_Ey_50_50_25.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'WireAirBox.fdtd_Point probe A_Ez_50_50_25.dat' using 1:2 every 1::2 with lines

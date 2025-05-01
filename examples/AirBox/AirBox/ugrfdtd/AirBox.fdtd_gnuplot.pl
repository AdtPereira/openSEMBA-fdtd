set term x11 persist           1
plot 'AirBox.fdtd_Point probe A_Ex_25_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'AirBox.fdtd_Point probe A_Ey_25_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'AirBox.fdtd_Point probe A_Ez_25_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           4
plot 'AirBox.fdtd_Point probe B_Ex_50_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           5
plot 'AirBox.fdtd_Point probe B_Ey_50_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           6
plot 'AirBox.fdtd_Point probe B_Ez_50_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           7
plot 'AirBox.fdtd_Point probe C_Hx_75_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           8
plot 'AirBox.fdtd_Point probe C_Hy_75_50_13.dat' using 1:2 every 1::2 with lines
set term x11 persist           9
plot 'AirBox.fdtd_Point probe C_Hz_75_50_13.dat' using 1:2 every 1::2 with lines

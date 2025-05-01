set term x11 persist           1
plot 'towelHanger.fdtd_wire_start_Wz_27_25_30_s1.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'towelHanger.fdtd_wire_end_Wz_43_25_30_s4.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'towelHanger.fdtd_wire_mid_Wx_35_25_32_s5.dat' using 1:2 every 1::2 with lines

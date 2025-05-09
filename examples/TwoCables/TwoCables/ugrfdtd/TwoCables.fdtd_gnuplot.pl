set term x11 persist           1
plot 'TwoCables.fdtd_Probe_Start_Ex_10_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'TwoCables.fdtd_Probe_Start_Ey_10_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'TwoCables.fdtd_Probe_Start_Ez_10_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           4
plot 'TwoCables.fdtd_Probe_End_Ex_50_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           5
plot 'TwoCables.fdtd_Probe_End_Ey_50_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           6
plot 'TwoCables.fdtd_Probe_End_Ez_50_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           7
plot 'TwoCables.fdtd_Wire probe_Wx_10_25_30_s3.dat' using 1:2 every 1::2 with lines
set term x11 persist           8
plot 'TwoCables.fdtd_Bulk probe_Jx_30_23_25__30_26_34.dat' using 1:2 every 1::2 with lines

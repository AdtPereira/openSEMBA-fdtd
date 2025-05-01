set term x11 persist           1
plot 'OneCable.fdtd_Probe_Start_Ex_10_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'OneCable.fdtd_Probe_Start_Ey_10_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'OneCable.fdtd_Probe_Start_Ez_10_25_10.dat' using 1:2 every 1::2 with lines
set term x11 persist           4
plot 'OneCable.fdtd_Bulk probe_Jx_30_23_25__30_26_34.dat' using 1:2 every 1::2 with lines

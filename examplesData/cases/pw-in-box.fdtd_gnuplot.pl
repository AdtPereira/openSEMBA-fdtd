set term x11 persist           1
plot 'pw-in-box.fdtd_before_Ex_3_3_1.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'pw-in-box.fdtd_inbox_Ex_3_3_3.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'pw-in-box.fdtd_after_Ex_3_3_5.dat' using 1:2 every 1::2 with lines

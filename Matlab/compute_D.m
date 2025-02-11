clear
close all

R_c = 3.3;
theta = pi/3; % lattice angle (60 = hexagonal)
D_min = R_c/sin(theta);
D_big = 2*R_c*cos(theta/2);

fprintf('D_min: %1.64f\nD_big: %1.64f\n', D_min, D_big)

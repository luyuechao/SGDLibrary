

clc; clear; close all;
%% set A and b
n = 1000;
%A = randi([0, 1],n, n)* randi([0, 1],n, n);
A = randn(n);
%A =  A * A';
%A = A' * A;
b = randn(n, 1);
w_init = randn(n, 1); 
tol = 1e-6;
x_matlab = pcg(A, b, tol);
x_my = rand_gd(A, b, tol);

fprintf('Error of matlab : %.6e\n', norm(A*x_matlab - b));
fprintf('Error of my: %.6e\n', norm(A*x_my - b));
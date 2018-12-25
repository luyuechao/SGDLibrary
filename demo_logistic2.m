function demo_logistic2()
% demonstration file for SGDLibrary.
%
% This file illustrates how to use this library in case of linear
% regression problem. This demonstrates SGD and SVRG algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 24, 2016
% Modified by H.Kasai on Nov. 03, 2016

clc; clear; close all;
run_me_first;
%% generate synthetic data        
% set number of dimensions
d = 100;
% set number of samples    
n = 65536;
% generate data
rho = 0.9; % convariance 
data = logistic_regression_data_generator(n, d, rho);
fprintf('condtition number of A : %10.6e\n', cond(data.x_train));
%% define problem definitions
problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test, 0); 

w_opt = problem.calc_solution(100, 'newton');
f_opt = problem.cost(w_opt);
fprintf('f_opt: %.24e\n', f_opt);

%% perform algorithms SGD and SVRG 
options.w_init = data.w_init;    
options.step_init = 0.01;
options.verbose = 2;
options.max_epoch = 50;
options.max_iter = 50;
options.f_opt = f_opt;
options.tol_optgap = 1e-10;
options.subsamp_hess_size = n;
options.step_alg = 'backtracking';
options.sub_mode = 'INEXACT'; %% inexact is PCG solver
%options.sub_mode = 'CHOLESKY';
[w_nt, info_nt] = newton(problem, options);

% options.subsamp_hess_size = double(int64(4 * d));
% options.sub_mode = 'RNS';
% [w_uni1, info_uni1] = subsamp_newton(problem, options);   
% options.sub_mode = 'LS';
% [w_ls1, info_ls1] = subsamp_newton(problem, options);

options.hessian_solver = 'CG';
options.subsamp_hess_size = double(int64(0.1 * n));
options.sub_mode = 'RNS';
[w_cg, info_cg] = subsamp_newton(problem, options);

options.hessian_solver = 'PCG';
[w_pcg, info_pcg] = subsamp_newton(problem, options);
%options.sub_mode = 'LS';
%[w_ls5, info_ls5] = subsamp_newton(problem, options);

algorithms = {'Newton', 'CG without precondition', 'Precondtioned CG'};
%w_list = {w_nt,w_uni1, w_ls1, w_uni5, w_ls5};
%info_list = {info_nt, info_uni1, info_ls1, info_uni5, info_ls5};
w_list = {w_nt, w_cg, w_pcg};
info_list = {info_nt, info_cg, info_pcg};

%% display cost/optimality gap vs number of gradient evaluations
display_graph('iter','cost', algorithms, w_list, info_list);
display_graph('iter','optimality_gap', algorithms, w_list, info_list);    
display_graph('flops', 'optimality_gap', algorithms, w_list, info_list);    

% display classification results
y_pred_list = cell(length(algorithms), 1);
accuracy_list = cell(length(algorithms), 1);    
for alg_idx=1:length(algorithms)
    if ~isempty(w_list{alg_idx})           
        p = problem.prediction(w_list{alg_idx});
        % calculate accuracy
        accuracy_list{alg_idx} = problem.accuracy(p); 

        fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, problem.accuracy(p));

        % convert from {1,-1} to {1,2}
        p(p==-1) = 2;
        p(p==1) = 1;
        % predict class
        y_pred_list{alg_idx} = p;
    else
        fprintf('Classificaiton accuracy: %s: Not supported\n', algorithms{alg_idx});
    end
end

% convert from {1,-1} to {1,2}
data.y_train(data.y_train==-1) = 2;
data.y_train(data.y_train==1) = 1;
data.y_test(data.y_test==-1) = 2;
data.y_test(data.y_test==1) = 1;  
%if plot_flag        
    display_classification_result(problem, algorithms, w_list, y_pred_list,...
             accuracy_list, data.x_train, data.y_train, data.x_test, data.y_test);    
%end    

end

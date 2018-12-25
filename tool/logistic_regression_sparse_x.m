function [data] = logistic_regression_data_generator(n, d, rho)
% Data generator for logistic regression problem.
%
% Inputs:
%       n               number of samples.
%       d               number of dimensions.
% Output:
%       data            data set
%       data.x_train    train data of x.
%       data.y_train    train data of y.
%       data.x_test     test data of x.
%       data.y_test     test data of y.
%       data.w_opt      solusion.
%
% This file is part of SGDLibrary.
%
% Created H.Kasai on Oct. 25, 2016

    % true
    w_opt = randn(d, 1); %% TODO change to sparse
    %w_opt = 0.5 * ones(d, 1);
    data.w_opt = w_opt;   

    % train data
    if exist('x0', 'var')
        cov_mat = repmat(rho, d, d) + diag(repmat(1-rho, d, 1));
        mu = zeros(n,d);
        x1 = mvnrand(mu, cov_mat)';
    else
        x1 = 20 * randn(d, n);
    end
    y1 = rand(1, n) < sigmoid(w_opt' * x1);
    y1 = 2*y1 - 1;
    assert(sum(y1 == 1) + sum(y1 == -1) == n);

    data.x_train = x1;
    data.y_train = y1;
    
    % test data    
    if exist('x0', 'var')
        cov_mat = repmat(rho, d, d) + diag(repmat(1-rho, d, 1));
        mu = zeros(n,d);
        x2 = mvnrand(mu, cov_mat)';
    else
        x2 = 20 * randn(d, n);
    end
    y2 = rand(1, n) < sigmoid(w_opt' * x2);
    y2 = 2*y2 - 1;
    assert(sum(y2 == 1) + sum(y2 == -1) == n);
    
    data.x_test = x2;
    data.y_test = y2;

    data.w_init = randn(d,1);
end


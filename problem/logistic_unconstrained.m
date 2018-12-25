classdef logistic_unconstrained
% This file defines logistic regression (binary classifier) problem class (version 2)
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%

%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 03, 2018


    properties
        name;    
        dim;
        samples;
        classes;  
        hessain_w_independent;
        d;
        n_train;
        n_test;
        x_train;
        y_train;
        x_test;
        y_test;
        x_norm;
        x;           
    end
    
    methods
        function obj = logistic_unconstrained(x_train, y_train, x_test, y_test, varargin)    
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;            

            obj.d = size(obj.x_train, 1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);      
            obj.name = 'logistic_regression_unconstrained';    
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.classes = 2;  
            obj.hessain_w_independent = false;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end

        function f = cost(obj, w)
			sigmod_result = sigmoid(obj.y_train.*(w'*obj.x_train));
			sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
			f = -sum(log(sigmod_result),2)/obj.n_train;        
        end
        
        function f = cost_batch(obj, w, indices)

            f = -sum(log(sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))/obj.n_train,2);

        end

        function g = grad(obj, w, indices)

            g = -sum(ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * (ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))',2)/length(indices);
            
        end

        function g = full_grad(obj, w)

            g = grad(obj, w, 1:obj.n_train);
        end

        function g = ind_grad(obj, w, indices)

            g = -ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * diag(ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))));

        end

        function h = hess(obj, w, indices)

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val);
          
            %h = 1/length(indices)* obj.x_train(:,indices) * diag(obj.y_train(indices).^2 .* c) * obj.x_train(:,indices)');
            B = bsxfun(@times,obj.x_train(:,indices), obj.y_train(indices).*sqrt(c));
            h = 1/length(indices) * (B*B');
        end

        function h = full_hess(obj, w)

            h = hess(obj, w, 1:obj.n_train);

        end

        function hv = hess_vec(obj, w, v, indices)

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            hv = 1/length(indices)* obj.x_train(:,indices) * diag(obj.y_train(indices).^2 .* c) * (obj.x_train(:,indices)' * v);

        end

        function p = prediction(obj, w)

            p = sigmoid(w' * obj.x_test);

            class1_idx = p>0.5;
            class2_idx = p<=0.5;         
            p(class1_idx) = 1;
            p(class2_idx) = -1;         

        end

        function a = accuracy(obj, y_pred)

            a = sum(y_pred == obj.y_test) / obj.n_test; 

        end

        function w_opt = calc_solution(obj, maxiter, method)

            if nargin < 3
                method = 'lbfgs';
            end        

            options.max_iter = maxiter;
            options.verbose = true;
            options.tol_optgap = 1.0e-24;
            options.tol_gnorm = 1.0e-16;
            options.step_alg = 'backtracking';

            if strcmp(method, 'sd')
                [w_opt,~] = sd(obj, options);
            elseif strcmp(method, 'cg')
                [w_opt,~] = ncg(obj, options);
            elseif strcmp(method, 'newton')
                options.sub_mode = 'INEXACT';    
                options.step_alg = 'non-backtracking'; 
                [w_opt,~] = newton(obj, options);
            else 
                options.step_alg = 'backtracking';  
                options.mem_size = 5;
                [w_opt,~] = lbfgs(obj, options);              
            end
        end
        
        %% for Sub-sampled Newton
        function h = diag_based_hess(obj, ~, indices, square_hess_diag)
            %X = obj.x_train(:,indices);
            %h = X * diag(square_hess_diag) * X' / length(indices);
            B = bsxfun(@times, obj.x_train(:,indices), sqrt(square_hess_diag'));
            h = 1/length(indices) * (B*B');
        end
        
        %% select diag Hessian of indices from n
        function square_hess_diag = calc_square_hess_diag(obj, w, indices)
            % H(x) = sigmoid(x).*(1-sigmoid(x));
            %hess_diag = 1./(1+exp(Y.*(X*w)))./(1+exp(-Y.*(X*w)));
            Xw = obj.x_train(:,indices)'*w;
            y = obj.y_train(indices)';
            yXw = y .* Xw;
            square_hess_diag = 1./(1+exp(yXw))./(1+exp(-yXw));
        end    

    end
end

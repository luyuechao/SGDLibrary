function [w, infos] = admm_sub_newton(problem, options)
% The alternating direction method of multipliers (ADMM) with sub_newton.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of GDLibrary and SGDLibrary.
%
% Originall code from
% https://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html.
% Originally modified by H.Kasai on Apr. 18, 2017
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  
    A = problem.x_train();
    b = problem.y_train();
    
    fprintf('A dim: (%d,%d) \n', size(A));
    fprintf('b dim: (%d,%d) \n', size(b));
 
    % extract options
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end
    
    if ~isfield(options, 'tol_gnorm')
        tol_gnorm = 1.0e-12;
    else
        tol_gnorm = options.tol_gnorm;
    end    
    
    if ~isfield(options, 'max_iter')
        max_iter = 100;
    else
        max_iter = options.max_iter;
    end 
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end   
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end    
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end  
    
    % augmented Lagrangian parameter
    if ~isfield(options, 'rho')
        rho = 1;
    else
        rho = options.rho;
    end  
    
    % over-relaxation parameter (typical values for alpha are between 1.0 and 1.8).
    if ~isfield(options, 'alpha')
        alpha = 1;
    else
        alpha = options.alpha;
    end
    
    % initialise
    iter = 0;
    w = zeros(d, 1);
    z = zeros(d, 1);
    u = zeros(d, 1); 
    
    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if ismethod(problem, 'reg')  
        infos.reg = problem.reg(w);   
    end    
    if store_w
        infos.w = w;       
    end
    
    % set start time
    start_time = tic();  
    
    % print info
    if verbose
        fprintf('ADMM Lasso: Iter = %03d, cost = %.6e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
    end

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        

        % update w
        w = update_w(problem, A, b, u, z, rho, w);

        % update z with relaxation
        zold = z;
        w_hat = alpha * w + (1 - alpha) * zold;
        z = problem.prox(w_hat + u, 1/rho);

        % update u
        u = u + (w_hat - z);            
        
        % calculate gradient
        grad = problem.full_grad(w);

        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count iter*n];      
        infos.optgap = [infos.optgap optgap];        
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if ismethod(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end        
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('ADMM Lasso: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end    
    
end

function w = update_w(problem, A, b, u, z, rho, w0)
    % solve the x update
    %   minimize [ -logistic(x_i) + (rho/2)||x_i - z^k + u^k||^2 ]
    % via Newton's method; for a single subsystem only.
    alpha = 0.1;
    BETA  = 0.5;
    TOLERANCE = 1e-5;
    MAX_ITER = 50;
    [d, n] = size(A);
    I = eye(d);
    if exist('x0', 'var')
        w = w0;
    else
        w = zeros(d, 1);
    end

    %f = @(w) (sum(log(1 + exp(C*w))) + (rho/2)*norm(w - z + u).^2);
    f = @(w) (-sum(log(sigmoid(b.*(w'*A))),2)/n + (rho/2)*norm(w - z + u).^2);
   
    for iter = 1:MAX_ITER
        fx = f(w);
        %g = C' * (exp(C*w)./(1 + exp(C*w))) + rho*(w - z + u);
        %H = C' * diag(exp(C*w)./(1 + exp(C*w)).^2) * C + rho*I;
        g = problem.full_grad(w) + rho*(w - z + u);
        H = problem.full_hess(w) + rho * I;
        dx = -H\g;   % Newton step
        dfx = g' * dx; % Newton decrement
        if abs(dfx) < TOLERANCE
            break;
        end
        % backtracking
        t = 1;
        while f(w + t*dx) > fx + alpha*t*dfx
            t = BETA*t;
        end
        w = w + t*dx;
    end
    
    fprintf('Iter = %d, ', iter);
end

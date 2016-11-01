function [w, infos] = gd(problem, options)
% Full gradient descent algorithm.
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
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Oct. 25, 2016


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  


    % extract options
    if ~isfield(options, 'step')
        step_init = 0.1;
    else
        step_init = options.step;
    end
    step = step_init;
    
    if ~isfield(options, 'step_alg')
        step_alg = 'backtracking';
    else
        step_alg  = options.step_alg;
    end  
   
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
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 1000;
    else
        max_epoch = options.max_epoch;
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
    
    if ~isfield(options, 'f_sol')
        f_sol = -Inf;
    else
        f_sol = options.f_sol;
    end    
    
    if ~isfield(options, 'store_sol')
        store_sol = false;
    else
        store_sol = options.store_sol;
    end    
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'STANDARD';
    else
        sub_mode = options.sub_mode;
    end  
    
    if ~isfield(options, 'S')
        S = eye(d);
    else
        S = options.S;
    end  
    
    ls_options.sub_mode = sub_mode;    
    if strcmp(sub_mode, 'STANDARD')
        %
    elseif strcmp(sub_mode, 'SCALING')
        ls_options.S = S;
    else
        %
    end    

    
    % initialise
    epoch = 0;
    
    % store first infos
    clear infos;
    infos.epoch = epoch;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_sol;
    infos.optgap = optgap;
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if store_sol
        infos.w = w;       
    end
    
    % set start time
    start_time = tic();    

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (epoch < max_epoch)        

        % line search
        if strcmp(step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, -grad, w, rho, c);
        elseif strcmp(step_alg, 'exact')
            step = exact_line_search(problem, 'GD', -grad, [], [], w, ls_options);
        else
        end
        
        % update w
        w = w - step * S * grad;
        
        % calculate gradient
        grad = problem.full_grad(w);

        % update epoch        
        epoch = epoch + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_sol;  
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.epoch = [infos.epoch epoch];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count epoch*n];      
        infos.optgap = [infos.optgap optgap];        
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if store_sol
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('GD: Epoch = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', epoch, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end    
    
end

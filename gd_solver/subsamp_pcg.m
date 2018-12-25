function [w, infos] = subsamp_pcg(problem, options)
% Sub-sampled Netwon method algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% Reference:
%       P. Xu, J. Yang, F. Ro-Khorasani, C. Re and M. W. Mahoney,
%       "Sub-sampled Newton Methods with Non-uniform Sampling,"
%       NIPS2016.
%
% This file is part of GDLibrary.
%
% Originally created by Peng Xu, Jiyan Yang on Feb. 20, 2016 (https://github.com/git-xp/Subsampled-Newton)
% Originally modified by H.Kasai on Mar. 16, 2017
% Modified by H.Kasai on Mar. 25, 2018
    

    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % extract options
    if ~isfield(options, 'step_init')
        step_init = 1;
    else
        step_init = options.step_init;
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
        w = randn(d, 1);
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
    
%     if ~isfield(options, 'step_init_alg')
%         % Do nothing
%     else
%         if strcmp(options.step_init_alg, 'bb_init')
%             % initialize by BB step-size
%             step_init = bb_init(problem, w);
%         end
%     end 
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'Uniform';
    else
        sub_mode = options.sub_mode;
    end

    if ~isfield(options, 'subsamp_hess_size')
        subsamp_hess_size = 200 * d;
    else
        subsamp_hess_size = options.subsamp_hess_size;
    end
 
   
    % initialise
    iter = 0;
    grad_count = 0;
    flops = 0;
    chol_flops = (d^3)/3 + (1/2 + 2)*d^2;
    hessian_flops = 2*subsamp_hess_size*d^2 + subsamp_hess_size * d;

    dirct = randn(d, 1);

    % store first infos
    clear infos;
    infos.w = w;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0; 
    infos.flops = 0;
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    pcg_iter = 0;
    % calculate gradient
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if ismethod(problem, 'reg')
        infos.reg = problem.reg(w);   
    end
    
    
    if strcmp(sub_mode, 'RNS')
        rnorms = problem.x_norm(); % square norm of each a (dim : n)
    end

    % set start time
    start_time = tic();

    if verbose
        fprintf('optgap = %5.3e, tol_optgap = %5.3e\n', optgap, tol_optgap);
        fprintf('Subsampled Newton (%s): Iter = %03d, cost = %.4e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
    end
    
    %hesfun = @(w) (problem.full_hess(w));
    fprintf('l = %d, n = %d, d = %d\n', problem.num_class, problem.samples, problem.d);
    
    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)
        
        % calculate gradient
        grad = problem.full_grad(w);    
        
        % step 2: calculate H*dirct = (-g)
        %tol = 1e-5; % relative tol for pcg
        tol = min(0.1/iter, norm(grad));
        fprintf('CG tol = %.4e\n',tol);
        %[dirct, pcg_iter] = problem.hv_pcg(w, -grad, dirct, tol);
        [dirct, pcg_iter] = problem.subSamp_hv_pcg(w, 1:n, -grad, dirct, tol);
        
        % linesearch
        rho = 1/2;
        c = 1e-4;
        step = backtracking_line_search(problem, dirct, w, rho, c);
        
        % update
        w = w + step * dirct;
        
        % proximal operator
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end
        
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
        infos.grad_calc_count = [infos.grad_calc_count grad_count];
        infos.flops = [infos.flops flops];
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
            fprintf('Cost = %.4e, gnorm = %.4e, optgap = %.4e, pcg_iter = %d\n', f_val, gnorm, optgap, pcg_iter);
        end        
    end %end of while
    
    fprintf('Total iteration : %d\n', iter);
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end    
    
end

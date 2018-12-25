function [x, iter_num] = rand_gd(A, b, rlav_tol, x0)
% Implementation of 
% An Introduction to the Conjugate Gradient Method 
% Without the Agonizing Pain.pdf
% Page 32
Precond = 1;
if Precond == 0
    n = numel(b);
    if nargin<3
        tol=1e-10;
        x = rand(n);
    elseif nargin < 4
        x = rand(n);
    elseif nargin == 4
        x = x0;
    else
        printf('paramter error');
    end
    r = b - A*x;
    v = r;
    iter_num = 1;
    tol_2 = (rlav_tol * norm(b))^2; %% squared tolerance
    
    for k = 1:n
        rTr = r'*r;
        if rTr < tol_2
            return
        end
        Av = A * v;
        alpha_i = rTr / (v'*Av);
        x = x + alpha_i * v;
        r = r - alpha_i * Av;
        beta_i = (r'*r) / rTr;
        v = r + beta_i * v;
        iter_num = iter_num + 1;
    end

else % Preconditioned CG
    pCn = (diag(A));
    
    n = numel(b);
    if nargin<3
        tol=1e-10;
        x = rand(n);
    elseif nargin < 4
        x = rand(n);
    elseif nargin == 4
        x = x0;
    else
        printf('paramter error');
    end
    r = b - A*x;
    v = r ./ pCn;
    iter_num = 1;
    tol_2 = (rlav_tol * norm(b))^2; %% squared tolerance
    
    for k = 1:n
        rTpr = r' * (r ./ pCn);
        if rTpr < tol_2
            return
        end
        Av = A * v;
        alpha_i = rTpr / (v' * Av);
        x = x + alpha_i * v;
        r = r - alpha_i * Av;
        pr = r ./ pCn;
        beta_i = (r'*r) / rTpr;
        v = pr + beta_i * v;
        iter_num = iter_num + 1;
    end    
end

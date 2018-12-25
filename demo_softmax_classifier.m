function  demo_softmax_classifier()
        
    clc; clear; close all;
    run_me_first;
    %% Set algorithms

    %algorithms = {'SD-STD', 'SD-EXACT', 'Newton-STD', 'BFGS'};
    algorithms = {'Sub-sampled PCG', 'L-BFGS-EXACT'};

    %% prepare dataset
    test_mode = 2;
    if test_mode == 0
        
%          n_per_class = 2;    % # of samples        
%          d = 5;                % # of dimensions     
%          l = 3;                % # of classes 
        n_per_class = 1000;    % # of samples        
        d = 300;               % # of dimensions     
        l = 10;                % # of classes 
        std = 0.85;            % standard deviation

        data = multiclass_data_generator(n_per_class, d, l, std);  
        n = length(data.y_train);
        d = d + 1; % adding '1' row for intersect
        
        % train data dim: (d+1) * n
        x_train =   [data.x_train; ones(1,n)];
        % transform class label into label logical matrix
        y_train = zeros(l, n);
        for j=1:n
            y_train(data.y_train(j),j) = 1;
        end

        % test data
        x_test = [data.x_test; ones(1,n)];
        % transform class label into label logical matrix
        y_test = zeros(l,n);
        for j=1:n
            y_test(data.y_test(j),j) = 1;
        end     
        
        lambda = 0.0001;
        w_opt = zeros(d*l,1);            
        
    elseif test_mode == 1
        % load real-world data
        data = importdata('./data/mnist/6000_data_0.001.mat');
        x_train = data.x_trn;
        x_train = normc(x_train); % normalize column of the data
        y_train = data.y_trn;
        x_test = data.x_tst;
        y_test = data.y_tst;       
        d = size(x_train,1);
        n = length(y_train);
        lambda = data.lambda;

        %w_opt = data.w_opt;
        
        l = data.L;
        w_opt = zeros(d*l,1);
    
    elseif test_mode == 2
        x_train = csvread('/Users/actionmask/project/normalized-data/gisette/gisette_train.data')';
        y_train_label = csvread('/Users/actionmask/project/normalized-data/gisette/gisette_train.labels01')';
        x_test = csvread('/Users/actionmask/project/normalized-data/gisette/gisette_valid.data')';
        y_test_label  = csvread('/Users/actionmask/project/normalized-data/gisette/gisette_valid.labels01')';
        l = 2; 
        [d, n] = size(x_train);
        y_train = zeros(l, n);
        for j = 1:n
            y_train(y_train_label(j)+1,j) = 1;
        end
        test_len = size(x_test, 2);
        y_test = zeros(l, test_len);
        for j = 1:test_len
            y_test(y_test_label(j)+1,j) = 1;
        end        
        lambda = 1e-3;
        w_opt = zeros(d*l,1);
    end

    fprintf('l = %d, d = %d, n = %d, lambda = %.3e\n', l, d, n, lambda);
    % set plot_flag
    if d > 4
        plot_flag = false;  % too high dimension  
    else
        plot_flag = true;
    end
    
    %% define problem definitions
    problem = softmax_regression(x_train, y_train, x_test, y_test, l, lambda);
   
    % initialize
    w_init = randn(d*l, 1);

    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    
    %% calculate solution
    if norm(w_opt)
    else
        % calculate solution
        w_opt = problem.calc_solution(300);
    end
    
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);   
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_iter = 100;
        options.verbose = true;  
        options.f_opt = f_opt;
        options.store_w = true;

        switch algorithms{alg_idx}
            case {'SD-STD'}
                
                options.step_alg = 'fix';
                options.step_init = 1;
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

            case {'SD-EXACT'}
                
                options.step_alg = 'exact';                
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);
                
            case {'Newton-STD'}
                
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);
                
            case {'Newton-CHOLESKY'}

                options.sub_mode = 'CHOLESKY';                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);
                
            case {'L-BFGS-EXACT'}
                
                options.step_alg = 'exact';    
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);
            
            case {'Sub-sampled PCG'} %%
              [w_list{alg_idx}, info_list{alg_idx}] = subsamp_pcg(problem, options);
              
            otherwise
                warn_str = [algorithms{alg_idx}, ' is not supported.'];
                warning(warn_str);
                w_list{alg_idx} = '';
                info_list{alg_idx} = '';                
        end
        
    end
    
    fprintf('\n\n');

    %% plot all
    close all;
    
    % display iter vs cost/gnorm
    display_graph('iter','cost', algorithms, w_list, info_list);
    %display_graph('iter','gnorm', algorithms, w_list, info_list);  
    display_graph('iter','optimality_gap', algorithms, w_list, info_list);
    
    % draw convergence sequence
    w_history = cell(1);
    cost_history = cell(1);    
    for alg_idx=1:length(algorithms)    
        w_history{alg_idx} = info_list{alg_idx}.w;
        cost_history{alg_idx} = info_list{alg_idx}.cost;
    end    
    draw_convergence_sequence(problem, w_opt, algorithms, w_history, cost_history);  
    
    % display classification results
    y_pred_list = cell(length(algorithms), 1);
    accuracy_list = cell(length(algorithms), 1);    
    for alg_idx=1:length(algorithms)    
        % predict class
        y_pred_list{alg_idx} = problem.prediction(w_list{alg_idx});
        % calculate accuracy
        accuracy_list{alg_idx} = problem.accuracy(y_pred_list{alg_idx}); 
        fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, accuracy_list{alg_idx});        
    end      

    % convert logial matrix to class label vector
    [~, y_train] = max(y_train, [], 1);
    [~, y_test] = max(y_test, [], 1);    
    if plot_flag
        display_classification_result(problem, algorithms, w_list, y_pred_list, accuracy_list, x_train, y_train, x_test, y_test);  
    end
    
end



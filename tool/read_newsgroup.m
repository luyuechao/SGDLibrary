function [data] = read_newsgroup(x_train, y_train, x_test, y_test)
% read news group data 
%
% Input data path
% Output:
%       data            data set
%       data.x_train    train data of x of size d x n*l.
%       data.y_train    train data of y of size l x n*l.
%       data.x_test     test data of x of size d x n*l.
%       data.y_test     test data of y of size l x n*l.
%
%   read data

    %d is the feature
%     [train_nnz, data_col] = size(x_train);
%     fprintf('Size of x_train : %d, %d\n', train_nnz, data_col);
%     fprintf('Size of y_train : %d, %d\n', length(y_train));
    % go through the sparse train data to get the shape of the data
    
    train_n = max(x_train(:, 1));
    train_d = max(x_train(:, 2));
    
    % go through the sparse test data to get the shape of the data
    test_n = max(x_test(:, 1));
    test_d = max(x_test(:, 2));
    
    %fprintf('train features: %d, test featrues: %d\n', train_d, test_d);
    feature_num = max(train_d, test_d);
    
    %fprintf('Dimension of Train: [%d, %d]\n', train_n, feature_num);
    %fprintf('Dimension of Test : [%d, %d]\n', test_n,  feature_num);
    
    % Read in the train data
    train_row_id = x_train(:, 1); % data #
    train_col_id = x_train(:, 2); % feature #
    train_val    = x_train(:, 3); % value
    
    data.y_train = sparse(1:length(y_train), y_train, ones(length(y_train), 1))';
    %fprintf('Size of data.y_train: %d, %d\n', size(data.y_train));
    
    test_label_num = size(y_test);
    if test_label_num ~= test_n
        fprintf('training data number %d  != label numer %d\n', test_label_num, test_n);
    end
    % Read in the test data
    test_row_id = x_test(:, 1);
    test_col_id = x_test(:, 2);
    test_val    = x_test(:, 3);
    
    data.y_test = sparse(1:length(y_test), y_test, ones(length(y_test), 1))';
    
    % savd data as d-by-n
    data.x_train = sparse(train_row_id, train_col_id, train_val)'; % transpose data
    data.x_test = sparse(test_row_id, test_col_id, test_val)';     % transpose data
    data.x_test = data.x_test(1:train_d, :); %% reduce 
    
end

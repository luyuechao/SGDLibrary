function [x_train, y_train_label, x_test, y_test_label] = read_cifar10

filePath = {'~/project/NewtonCG/cifar-10-batches-mat'};

files = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat'};

x_train = [];
y_train_label = [];

for i=1:5
    load(fullfile(char(filePath), char(files(i))))
    x_train = [x_train; data];
    y_train_label = [y_train_label; labels];
end

% load test
load(fullfile(char(filePath), char('test_batch.mat')));
x_test = data;
y_test_label = labels;

x_train = double(x_train'); % transpose
bsxfun(@minus, x_train, mean(x_train, 2)); % substract mean
x_train = normc(x_train);

x_test = double(x_test'); % transpose
bsxfun(@minus, x_test, mean(x_test, 2)); % substract mean 
x_test = normc(x_test);

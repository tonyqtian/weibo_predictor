%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%  Featured data loaded into memory

%  x0:  constant

%  x1:  count of user tweets

%  x2:  sum of interset of user keywords and item keywords
%  x3:  count of interset of user keywords and item keywords

%  x4:  sum of first teer user which a user @ed (weighted with commuting number)
%  x5:  sum of first teer user which a user retweeted (weighted with commuting number)
%  x6:  sum of first teer user which a user commented (weighted with commuting number)

%  x7:  count of first teer user which a user @ed
%  x8:  count of first teer user which a user retweeted
%  x9:  count of first teer user which a user commented

%  x10:  count of first teer user which a user followed

%  x11:  sum of first teer user which a user @ed (weighted with commuting number) who contain the target item
%  x12:  sum of first teer user which a user retweeted (weighted with commuting number) who contain the target item
%  x13:  sum of first teer user which a user commented (weighted with commuting number) who contain the target item

%  x14:  count of first teer user which a user @ed who contain the target item
%  x15:  count of first teer user which a user retweeted who contain the target item
%  x16:  count of first teer user which a user commented who contain the target item

%  x17:  count of first teer user which a user followed who contain the target item


%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

fprintf('Loading train file...\n');
data = csvread('train_log_demo.csv');
%X_raw = data(:, [4:18]); y = data(:, 19);
X = data(:, [1:18]); y = data(:, 19);
clear data;
fprintf('Train file loaded. \nLogprocessing...\n');
%X = log_norm(X_raw, 0);
%X = X_raw;
%clear X_raw;
fprintf('Training data prepared.\n');


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
%X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

fprintf('\nStarting Regularization...\n');

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

fprintf('\nRegularization finished.\n');
theta' %'

clear X;
clear y;
fprintf('Loading test file...\n');
data = csvread('test_log_demo.csv');
%X_raw = data(:, [4:18]); y = data(:, 19);
X = data(:, [1:18]); y = data(:, 19);
clear data;
fprintf('Test file loaded. \nLogrocessing...\n');
%X = log_norm(X_raw, 0);
%X = X_raw;
%clear X_raw;
fprintf('Computung prediction...\n');
% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Testing data prepared.\n');

fprintf('Recall: %d  %.02f%%\n', sum(p), sum(p)/size(p,1) * 100);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
clear p;
clear y;

result = sigmoid(X * theta);
%clear X;
fprintf('\nStart saving result...\n');
csvwrite('test_full_y.csv', result);
fprintf('All finished.\n');
%% Initialization
clear ; close all; clc;

load('data/X.mat');
load('data/X_sorted_indices.mat');

X = X/100;
y = X_sorted_indices/10;


m = size(X, 1);

input_layer_size  = 10;  % 10 numbers
hidden_layer_size_1 = 50;   % hidden units
%hidden_layer_size_2 = 50
num_labels = 10;          % 10 sorted numbers

% Weight regularization parameter
lambda = 0;

% Unroll parameters 
Theta1 = randInitializeWeights(input_layer_size + 1, hidden_layer_size_1);
Theta2 = randInitializeWeights(hidden_layer_size_1 + 1, hidden_layer_size_2);
%Theta3 = randInitializeWeights(hidden_layer_size_2 + 1, num_labels);
initial_nn_params = [Theta1(:) ; Theta2(:)];               
             


%forward prop test
%predict(Theta1,Theta2,X,num_labels)


%Cost function test
%[J grad] = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
%                   num_labels, X, y, lambda)

%implement fmincg here


% Create short hand for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size_1, ...
                                   num_labels, X, y, lambda);
                                   
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size_1 * (input_layer_size + 1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size_1 * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size_1 + 1));

%Check that the new theta is classifying correctly by testing with forward prop (predict)
%testset = [9,8,7,6,5,1,2,3,4,5]*10/100
%testset2 = [1,2,3,4,5,6,7,8,9,9]/10
%predict(Theta1,Theta2,testset,num_labels)
%predict(Theta1,Theta2,testset2,num_labels)


result = assessSigmoid(X,Theta1,Theta2,X,m,num_labels)

%gradient checking

%celebrate


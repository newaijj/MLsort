%% Initialization
clear ; close all; clc;

load X; load X_sorted; load X_sorted_indices;

m = size(X, 1);

input_layer_size  = 100;  % 100 numbers
hidden_layer_size = 100;   % 100 hidden units
num_labels = 100;          % 100 sorted numbers

% Weight regularization parameter
lambda = 1;

% Unroll parameters 
Theta1 = randInitializeWeights(input_layer_size + 1, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size + 1, num_labels);
initial_nn_params = [Theta1(:) ; Theta2(:)];               
             


%forward prop test
%predict(Theta1,Theta2,X,num_labels)


%Cost function test
%[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                   num_labels, X, X_sorted_indices, lambda);




%implement fmincg here

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, X_sorted_indices, lambda);
                                   
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%Check that the new theta is classifying correctly by testing with forward prop (predict)

%gradient checking

%celebrate

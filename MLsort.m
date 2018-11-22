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
nn_params = [Theta1(:) ; Theta2(:)];               
                 
%forward prop test
%predict(Theta1,Theta2,X,num_labels)


%check that this is computing correctly
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, X_sorted_indices, lambda)
                   
%implement fmincg here

%Check that the new theta is classifying correctly by testing with forward prop (predict)

%celebrate

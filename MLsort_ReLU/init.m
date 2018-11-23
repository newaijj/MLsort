%% Initialization
clear ; close all; clc


%load dataset

fprintf('Load everything  \n');
load('data/X.mat');
load('data/X_sorted.mat');
load('data/X_CV.mat');
load('data/X_CV_sorted.mat');
load('data/X_test.mat');
load('data/X_test_sorted.mat');

%% Setup the parameters you will use for this exercise
fprintf('Setup perimeters \n');
input_layer_size  = size(X, 2);
hidden_layer_size = 20;
output_layer_size = input_layer_size;

fprintf('Initialise theta \n');
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);
%Theta1 and Theta2 are matrices of size(output, input)
%multiply with X using Theta*X to get output of size(output, 1) which is the nodes

%unroll Parameters
fprintf('Unroll parameters \n');
initial_nn_params = [Theta1(:) ; Theta2(:)];

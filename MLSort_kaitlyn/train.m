lambda = 0;
max_iter = 500;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   output_layer_size, X, X_sorted, lambda);

fprintf(['Cost at initial parameters: %f \n'], J);

% fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
% checkNNGradients;

options = optimset('MaxIter', max_iter);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X, X_sorted, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

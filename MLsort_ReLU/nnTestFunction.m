function J = nnTestFunction(nn_params, ...
                            input_layer_size, ...
                            hidden_layer_size, ...
                            output_layer_size, ...
                            X, X_sorted)

% basically nnCostFunction but without lambda, for evaluation

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                output_layer_size, (hidden_layer_size + 1));

%size of input
m = size(X, 1);

%initialise J and gradients
J = 0;

%forward propagation:
%z2, a2: unactivated, activated nodes in hidden layer
%z3, a3: unactivated, activated nodes in output layer

z2 = [ones(m,1) X]*Theta1';
a2 = relu(z2);
z3 = [ones(m, 1) a2]*Theta2';
a3 = relu(z3);

J = 1/(2*m)*sum(sum((a3 - X_sorted).^2))

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, ...
                                   X_train, X_train_sorted, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                output_layer_size, (hidden_layer_size + 1));

%size of input
m = size(X_train, 1);

%initialise J and gradients
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%forward propagation:
%z2, a2: unactivated, activated nodes in hidden layer
%z3, a3: unactivated, activated nodes in output layer

z2 = [ones(m,1) X_train]*Theta1';
a2 = relu(z2);
z3 = [ones(m, 1) a2]*Theta2';
a3 = relu(z3);

J = 1/(2*m)*sum(sum((a3 - X_train_sorted).^2))
J = J + lambda/(2*m)*(sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2)));

err_3 = a3 - X_train_sorted;
err_2 = err_3*Theta2.*[ones(m, 1) reluGradient(z2)];

Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for i=1:m
  Delta2 = Delta2 + err_3(i, :)'*[1 a2(i, :)];
  Delta1 = Delta1 + err_2(i, 2:end)'*[1 X_train(i, :)];
end

Theta1_grad = 1/m*Delta1;
Theta2_grad = 1/m*Delta2;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m*Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m*Theta2(:, 2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

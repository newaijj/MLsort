function prediction = predict(nn_params, ...
                                input_layer_size, ...
                                hidden_layer_size, ...
                                output_layer_size, ...
                                X, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                            hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                            output_layer_size, (hidden_layer_size + 1));

%size of input
m = size(X, 1);

z2 = [ones(m,1) X]*Theta1';
a2 = relu(z2);
z3 = [ones(m, 1) a2]*Theta2';
a3 = relu(z3);

J = 1/(2*m)*sum(sum((a3 - X_sorted).^2))

fprintf('prediction before rounding: \n');
a3

prediction = zeros(size(a3));
for i = 1:size(a3, 1)
  for j = 1:size(a3, 2)
    [min_value indices] = min(abs( X(i, :)-a3(i, j) ));
    prediction(i, j) = X(i, indices);
  end
end

fprintf('prediction after rounding: \n');
prediction

end

% evaluate number of hidden units
lambda = 0;
max_iter = 500;
min_units = input_layer_size
max_units = input_layer_size + 90
mult = 10

error_train = zeros(size(1, (max_units - min_units + mult)/mult));
error_val = zeros(size(1, (max_units - min_units + mult)/mult));

for i = 1:(max_units - min_units + mult)/mult
  hidden_layer_size = input_layer_size + (i-1)*mult;
  m = size(X, 1);
  Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);
  initial_nn_params = [Theta1(:) ; Theta2(:)];

  options = optimset('MaxIter', max_iter);
  costFunction = @(p) nnCostFunction(p, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     output_layer_size, ...
                                     X, X_sorted, lambda);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   output_layer_size, (hidden_layer_size + 1));

  error_train(i) = nnTestFunction(nn_params, input_layer_size, hidden_layer_size, ...
                      output_layer_size, X, X_sorted);

  error_val(i) = nnTestFunction(nn_params, input_layer_size, hidden_layer_size, ...
                      output_layer_size, X_CV, X_CV_sorted);

end

plot(min_units:mult:max_units, error_train, min_units:mult:max_units, error_val);
title('Learning curve with varying hidden units')
legend('Train', 'Cross Validation')
xlabel('Number of hidden units')
ylabel('Error')

[min_err indices] = min(error_val);
hidden_layer_size = input_layer_size + (indices - 1)*mult

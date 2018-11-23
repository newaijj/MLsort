%evaluate lambda
max_iter = 500;

lambda_list = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05];

error_train = zeros(size(lambda_list));
error_val = zeros(size(lambda_list));

for i = 1:size(lambda_list, 2)
  lambda = lambda_list(i);
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

plot(lambda_list, error_train, lambda_list, error_val);
title('Learning curve with varying lambda')
legend('Train', 'Cross Validation')
xlabel('Value of lambda')
ylabel('Error')

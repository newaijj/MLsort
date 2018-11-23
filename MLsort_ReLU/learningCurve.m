function [error_train, error_val] = ...
        learningCurve(input_layer_size, ...
                        hidden_layer_size, ...
                        output_layer_size, ...
                        X, X_sorted, X_CV, X_CV_sorted, lambda, max_iter, mult)

m = size(X, 1);

error_train = zeros(m/mult, 1);
error_val   = zeros(m/mult, 1);

for i = 1 : m/mult
  j = i*mult;
  X_train = X(1:j, :);
  X_train_sorted = X_sorted(1:j, :);

  Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);
  initial_nn_params = [Theta1(:) ; Theta2(:)];

  options = optimset('MaxIter', max_iter);
  costFunction = @(p) nnCostFunction(p, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     output_layer_size, ...
                                     X_train, X_train_sorted, lambda);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   output_layer_size, (hidden_layer_size + 1));

  error_train(i) = nnTestFunction(nn_params, input_layer_size, hidden_layer_size, ...
                     output_layer_size, X_train, X_train_sorted);

  error_val(i) = nnTestFunction(nn_params, input_layer_size, hidden_layer_size, ...
                     output_layer_size, X_CV, X_CV_sorted);

end

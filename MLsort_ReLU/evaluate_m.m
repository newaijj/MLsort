% evaluate m
lambda = 0.05;
max_iter = 1000;
mult = 50;
hidden_layer_size = 40;

[error_train, error_val] = ...
    learningCurve(input_layer_size, hidden_layer_size, ...
                  output_layer_size, X, X_sorted, ...
                  X_CV, X_CV_sorted, ...
                  lambda, max_iter, mult);

plot(1:size(error_train), error_train, 1:size(error_val), error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel(sprintf('Number of training examples (x %f)', mult))
ylabel('Error')

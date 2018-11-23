% evaluate m
lambda = 0;
max_iter = 150;
mult = 100;

[error_train, error_val] = ...
    learningCurve(input_layer_size, hidden_layer_size, ...
                  output_layer_size, X, X_sorted, ...
                  X_CV, X_CV_sorted, ...
                  lambda, max_iter, mult);

plot(1:size(error_train), error_train, 1:size(error_val), error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel(sprintf('Number of training examples (x %f)', mult))
ylabel('Error')

%evaluate lambda
lambda_list = [0, 0.3, 3, 10, 20];

%evaluate no. of hidden units

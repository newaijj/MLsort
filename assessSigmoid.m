function accuracy = assessSigmoid(X_unsorted,Theta1,Theta2,testset,m,num_labels)

predictedOrders = predict(Theta1,Theta2,testset,num_labels);

successes = 0;
for i = 1:m
  [dummy, index_prediction] = sort(predictedOrders(i,:));
  [dummy, index_X] = sort(X_unsorted(i,:));
  if (index_prediction == index_X)
    successes = successes + 1;
  endif
endfor
accuracy = successes/m;

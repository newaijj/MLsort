function accuracy = assessSigmoid(X_unsorted,Theta1,Theta2,testset,m,num_labels)

predictedOrders = predict(Theta1,Theta2,testset,num_labels);

successes = 0;

%generating predicted indices from predicted orders
predictedIndices = zeros(size(predictedOrders));
for i = 1:m
  holder = sort(predictedOrders(i,:),2);
  for j = 1:num_labels
    predictedIndices(i,j) = min(find(holder == predictedOrders(i,j)));
  end
end
  

for i = 1:m
  [dummy, index_X] = sort(X_unsorted(i,:));
  if (predictedIndices(i,:) == index_X)
    successes = successes + 1;
  end
end

accuracy = successes/m;

end
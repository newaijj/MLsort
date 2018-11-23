function [X X_sorted X_sorted_indices] = generateData(num_lists,list_size,num_min,num_max)

X = floor(rand(num_lists,list_size) .* (num_max - num_min) .+ num_min);
X_sorted = zeros(size(X,1),size(X,2));
[X_sorted, indices] = sort(X,2);
X_sorted_indices = indices/list_size;

end

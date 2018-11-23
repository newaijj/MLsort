function g = relu(z)

g = zeros(size(z));
g = z .* (z > 0) + 0.01 * z .* (z <= 0);

end

function g = reluGradient(z)

g = zeros(size(z));
g = 1 * (z > 0) + 0.01 * (z <= 0);

end

function [V, gradV] = eval_expansion(X, NN, c, a)
%EVAL_EXPANSION  Value and gradient of V(x) = sum_j c_j phi_j(x).
%
%   [V, gradV] = EVAL_EXPANSION(X, NN, c, a) returns V (N-by-1) and
%   gradV (N-by-d) for the tensor Legendre basis of TENSOR_LEGENDRE_BASIS.

[N, d] = size(X);
n = size(NN, 1);
c = c(:);
a = a(:).' .* ones(1, d);
s = prod(1 ./ sqrt(a));

[L, dL] = legendre_tables(X ./ a, max(NN(:)));

Phi = s * ones(N, n);
for k = 1:d
    Phi = Phi .* reshape(L(:, k, NN(:, k) + 1), N, n);
end
V = Phi * c;

if nargout > 1
    gradV = zeros(N, d);
    for k = 1:d
        cols = find(NN(:, k) > 0 & c ~= 0);
        m = numel(cols);
        if m == 0, continue; end
        Gk = (s / a(k)) * reshape(dL(:, k, NN(cols, k) + 1), N, m);
        for kk = [1:k-1, k+1:d]
            Gk = Gk .* reshape(L(:, kk, NN(cols, kk) + 1), N, m);
        end
        gradV(:, k) = Gk * c(cols);
    end
end
end

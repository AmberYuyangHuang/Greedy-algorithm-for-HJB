function [Phi, G] = tensor_legendre_basis(X, NN, a)
%TENSOR_LEGENDRE_BASIS  Tensor-product Legendre basis on a box and its gradient.
%
%   [Phi, G] = TENSOR_LEGENDRE_BASIS(X, NN, a) evaluates the basis functions
%       phi_j(x) = prod_k p_{NN(j,k)}(x_k / a_k) / sqrt(prod_k a_k),
%   which are orthonormal in L^2 of the box prod_k [-a_k, a_k].
%
%   Inputs
%     X   N-by-d evaluation points
%     NN  n-by-d multi-indices (one row per basis function)
%     a   half-widths of the box (scalar or 1-by-d)
%
%   Outputs
%     Phi N-by-n values,     Phi(i,j)   = phi_j(X(i,:))
%     G   N-by-d-by-n grads, G(i,k,j)   = d phi_j / d x_k at X(i,:)
 

[N, d] = size(X);
n = size(NN, 1);
a = a(:).' .* ones(1, d);
s = prod(1 ./ sqrt(a));

[L, dL] = legendre_tables(X ./ a, max(NN(:)));

Phi = s * ones(N, n);
for k = 1:d
    Phi = Phi .* reshape(L(:, k, NN(:, k) + 1), N, n);
end

if nargout > 1
    G = zeros(N, d, n);
    for k = 1:d
        cols = find(NN(:, k) > 0);          % d phi_j / d x_k = 0 otherwise
        m = numel(cols);
        if m == 0, continue; end
        Gk = (s / a(k)) * reshape(dL(:, k, NN(cols, k) + 1), N, m);
        for kk = [1:k-1, k+1:d]
            Gk = Gk .* reshape(L(:, kk, NN(cols, kk) + 1), N, m);
        end
        G(:, k, cols) = reshape(Gk, N, 1, m);
    end
end
end

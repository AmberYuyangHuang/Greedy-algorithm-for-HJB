function [X, w] = gauss_legendre(n, a)
%GAUSS_LEGENDRE  Tensor Gauss-Legendre rule on the box prod_k [-a_k, a_k].
%
%   [X, w] = GAUSS_LEGENDRE(n, a) returns the n^d nodes X (n^d-by-d) and
%   weights w (n^d-by-1) of the tensor-product Gauss-Legendre rule with n
%   points per direction; d = numel(a). Nodes and weights of the 1-D rule
%   are computed with the Golub-Welsch algorithm.

k = 1:n-1;
beta = k ./ sqrt(4 * k.^2 - 1);
[Vec, D] = eig(diag(beta, 1) + diag(beta, -1));
[x1, idx] = sort(diag(D));
w1 = 2 * Vec(1, idx).'.^2;

d = numel(a);
nodes = cell(1, d);
wts   = cell(1, d);
for j = 1:d
    nodes{j} = a(j) * x1;
    wts{j}   = a(j) * w1;
end
[nodes{:}] = ndgrid(nodes{:});
[wts{:}]   = ndgrid(wts{:});

X = zeros(n^d, d);
w = ones(n^d, 1);
for j = 1:d
    X(:, j) = nodes{j}(:);
    w = w .* wts{j}(:);
end
end

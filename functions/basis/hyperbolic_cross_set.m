function NN = hyperbolic_cross_set(d, J)
%HYPERBOLIC_CROSS_SET  Hyperbolic cross multi-indices with index J.
%
%   NN = HYPERBOLIC_CROSS_SET(d, J) returns all n in N_0^d with
%       prod_k (n_k + 1) <= 2^J + 1,
%   one multi-index per row. The constant (all-zero) index comes first and
%   the remaining rows are ordered by total degree, then lexicographically.
%
%   Sizes used in the paper: d = 2, J = 4 -> 52;  d = 6, J = 4 -> 537;
%   d = 35, J = 3 -> 10396.

lim = 2^J + 1;
NN = zeros(1, 0);                       % a single empty partial index
for k = 1:d
    % append the k-th coordinate to every admissible partial index
    prods = prod(NN + 1, 2);
    next = zeros(0, k);
    for n = 0:lim-1
        keep = prods * (n + 1) <= lim;
        if ~any(keep), break; end
        next = [next; NN(keep, :), n * ones(nnz(keep), 1)]; %#ok<AGROW>
    end
    NN = next;
end

NN = sortrows([sum(NN, 2), NN]);
NN = NN(:, 2:end);
end

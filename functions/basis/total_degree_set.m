function NN = total_degree_set(d, M)
%TOTAL_DEGREE_SET  Total-degree multi-indices of degree at most M.
%
%   NN = TOTAL_DEGREE_SET(d, M) returns all n in N_0^d with sum(n) <= M, one
%   per row, with the constant index first. Rows are enumerated with the
%   first coordinate running fastest.
%
%   Sizes used in the paper: d = 1, M = 10 -> 11;  d = 2, M = 8 -> 45.

grids = cell(1, d);
[grids{:}] = ndgrid(0:M);
NN = zeros(numel(grids{1}), d);
for k = 1:d
    NN(:, k) = grids{k}(:);
end
NN = NN(sum(NN, 2) <= M, :);
end

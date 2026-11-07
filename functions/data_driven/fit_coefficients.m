function c = fit_coefficients(Phi, G, z, g, lambda, reltol)
%FIT_COEFFICIENTS  Gradient-augmented regression (Section 3.3).
%
%   c = FIT_COEFFICIENTS(Phi, G, z, g, lambda) fits V = sum_j c_j phi_j to
%   values z (N-by-1) and gradients g (N-by-d) by solving
%       min_c (1/(2 N_r)) || X c - b ||^2 + lambda ||c||_1,
%       X = [Phi; G],  b = [z; g],
%   with N_r = N (1 + d) rows. This is the normalisation of MATLAB's LASSO,
%   so lambda here is the value quoted in the paper. Phi is N-by-n and G is
%   N-by-d-by-n (from TENSOR_LEGENDRE_BASIS). lambda = 0 gives ordinary
%   least squares.
 

if nargin < 6 || isempty(reltol), reltol = 1e-8; end

[N, n] = size(Phi);
d = size(g, 2);
F = [Phi; reshape(G, N * d, n)];
y = [z; g(:)];

if lambda == 0
    c = F \ y;
else
    c = lasso(F, y, 'Lambda', lambda, 'Standardize', false, ...
              'Intercept', false, 'MaxIter', 1e10, 'RelTol', reltol);
end
end

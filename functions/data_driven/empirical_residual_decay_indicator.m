function e = empirical_residual_decay_indicator(G_c, gradV, f, ell, beta)
%EMPIRICAL_RESIDUAL_DECAY_INDICATOR  Empirical residual-decay indicator (27).
%
%   e = EMPIRICAL_RESIDUAL_DECAY_INDICATOR(G_c, gradV, f, ell, beta) returns,
%   for every candidate basis function j, the empirical residual-decay
%   indicator of the stationary HJB equation (2) with q = I on the active
%   samples x_1, ..., x_N (uniform weights mu_i = 1):
%       R_i          = -1/(4 beta) |grad V(x_i)|^2 + grad V(x_i) . f(x_i) + ell(x_i)   (22)
%       a_k(x_i)     = f(x_i) - 1/(2 beta) grad V(x_i)                                  (23)
%       L_k phi_j(x_i) = a_k(x_i) . grad phi_j(x_i)
%       e_j          = | sum_i R_i L_k phi_j(x_i) | / ( sum_i |L_k phi_j(x_i)|^2 )^(1/2),
%   with e_j = 0 when the denominator vanishes (e.g. constant phi_j).
%
%   G_c    N-by-d-by-nc candidate gradients (from TENSOR_LEGENDRE_BASIS)
%   gradV  N-by-d gradient of the current approximation at the samples
%   f      N-by-d drift at the samples,  ell  N-by-1 running cost,
%   beta   control weight in (2).

[N, ~, nc] = size(G_c);
R = -sum(gradV.^2, 2) / (4 * beta) + sum(gradV .* f, 2) + ell;    % N-by-1
a = f - gradV / (2 * beta);                                       % N-by-d
L_phi = reshape(sum(G_c .* a, 2), N, nc);                         % N-by-nc

num = L_phi' * R;
den = sqrt(sum(L_phi.^2, 1))';

e = zeros(nc, 1);
ok = den > 0;
e(ok) = abs(num(ok)) ./ den(ok);
end

function e = supervised_loss_indicator(Phi_c, G_c, res_f, res_g, omega)
%SUPERVISED_LOSS_INDICATOR  Supervised-loss basis indicator (26).
%
%   e = SUPERVISED_LOSS_INDICATOR(Phi_c, G_c, res_f, res_g, omega) returns,
%   for every candidate basis function j, with uniform weights mu_i = 1,
%       e_j = | z_j' h | / || z_j ||,
%       z_j = [sqrt(w1) phi_j(X); sqrt(w2) grad phi_j(X)],
%       h   = [sqrt(w1) res_f;    sqrt(w2) res_g],
%   i.e. the decrease of the weighted supervised loss obtained by adding
%   phi_j alone, with an optimal coefficient, to the current fit.
%
%   Phi_c  N-by-nc candidate values, G_c N-by-d-by-nc candidate gradients,
%   res_f  N-by-1 value residuals v_i - V(x_i),
%   res_g  N-by-d gradient residuals g_i - grad V(x_i),
%   omega  [w1, w2] = [omega_1, omega_2].

[N, nc] = size(Phi_c);
d = size(res_g, 2);
Gc = reshape(G_c, N * d, nc);

num = omega(1) * (Phi_c' * res_f) + omega(2) * (Gc' * res_g(:));
den = sqrt(omega(1) * sum(Phi_c.^2, 1)' + omega(2) * sum(Gc.^2, 1)');

e = zeros(nc, 1);
ok = den > 0;
e(ok) = abs(num(ok)) ./ den(ok);
end

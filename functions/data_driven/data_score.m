function s = data_score(X, z, g, NN_sel, c, a, omega)
%DATA_SCORE  Gradient-augmented data score (28).
%
%   s = DATA_SCORE(X, z, g, NN_sel, c, a, omega) returns, for every
%   candidate sample (X(i,:), z(i), g(i,:)),
%       score_i = omega(1) (z_i - V(x_i))^2 + omega(2) || g_i - grad V(x_i) ||^2,
%   where V is the current approximation with coefficients c on the basis
%   NN_sel. omega defaults to [1, 1].

if nargin < 7, omega = [1, 1]; end
[V, gradV] = eval_expansion(X, NN_sel, c, a);
s = omega(1) * (z - V).^2 + omega(2) * sum((g - gradV).^2, 2);
end

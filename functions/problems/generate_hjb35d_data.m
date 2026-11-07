function D = generate_hjb35d_data()
%GENERATE_HJB35D_DATA  Data for the 35-dimensional test of Section 6.5.
%
%   D = GENERATE_HJB35D_DATA() returns 20000 training and 5000 validation
%   samples in the format of LOAD_DATASET (fields X, z, g, Xv, zv, gv, d, a),
%   plus the drift f, the running cost ell at the training samples and beta,
%   which the empirical residual-decay indicator (27) needs.
%   Nothing is written to disk; FIG6_HJB35D_DATA_DRIVEN calls it directly.
%
%   Stationary HJB equation (2) on [-1, 1]^d, d = 35, beta = 1, q = I:
%       -1/(4 beta) |grad V|^2 + grad V . f + ell = 0,
%   with drift f_i(x) = -kappa_i x_i, kappa_i = 0.3 + 0.5 (i-1)/(d-1).
%   The value function is prescribed and depends on the first s = 6
%   coordinates only,
%       V(x) = 1 + sum_{i<=s} (rho_i x_i^2 + eta_i x_i^3) + 0.08 sum_{i<s} x_i x_{i+1},
%   rho_i = 1.2 - 0.13 (i-1),  eta_i = 0.06 (1 - 0.11 (i-1)),  i = 1, ..., s,
%   and the running cost is ell = 1/(4 beta) |grad V|^2 - grad V . f, so
%   that V solves the HJB equation exactly. States are sampled uniformly
%   (rng(1)).

rng(1);

d = 35;
beta = 1;
s = 6;
N_train = 20000;
N_valid = 5000;

kappa = linspace(0.3, 0.8, d);
params.rho = zeros(1, d);   params.rho(1:s) = linspace(1.2, 0.55, s);
params.eta = zeros(1, d);   params.eta(1:s) = 0.06 * linspace(1.0, 0.45, s);
params.pair_i = 1:s-1;
params.pair_j = 2:s;
params.pair_coef = 0.08 * ones(1, s - 1);

X_train = 2 * rand(N_train, d) - 1;
X_valid = 2 * rand(N_valid, d) - 1;

[V_train, grad_train, ell_train, f_train] = exact_solution(X_train, beta, kappa, params);
[V_valid, grad_valid] = exact_solution(X_valid, beta, kappa, params);

D.d = d;
D.a = ones(1, d);
D.X = X_train;   D.z  = V_train;   D.g  = grad_train;
D.Xv = X_valid;  D.zv = V_valid;   D.gv = grad_valid;
D.f = f_train;   D.ell = ell_train;   D.beta = beta;     % for indicator (27)

res_train = -sum(grad_train.^2, 2) / (4 * beta) + sum(grad_train .* f_train, 2) + ell_train;
fprintf('Generated %d training and %d validation samples (max |HJB residual| = %.2e)\n', ...
        N_train, N_valid, max(abs(res_train)));
end


function [V, gradV, ell, f] = exact_solution(X, beta, kappa, params)
N = size(X, 1);
V = ones(N, 1);
gradV = zeros(size(X));
for i = find(params.rho ~= 0 | params.eta ~= 0)
    xi = X(:, i);
    V = V + params.rho(i) * xi.^2 + params.eta(i) * xi.^3;
    gradV(:, i) = gradV(:, i) + 2 * params.rho(i) * xi + 3 * params.eta(i) * xi.^2;
end
for p = 1:numel(params.pair_i)
    i = params.pair_i(p);  j = params.pair_j(p);  w = params.pair_coef(p);
    V = V + w * X(:, i) .* X(:, j);
    gradV(:, i) = gradV(:, i) + w * X(:, j);
    gradV(:, j) = gradV(:, j) + w * X(:, i);
end
f = -X .* kappa;
ell = sum(gradV.^2, 2) / (4 * beta) - sum(gradV .* f, 2);
end

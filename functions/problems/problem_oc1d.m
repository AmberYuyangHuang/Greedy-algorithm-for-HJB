function p = problem_oc1d()
%PROBLEM_OC1D  One-dimensional optimal control problem of Section 6.2.
%
%   min_u int_0^T (1/4)(y^2 e^y + 2 y e^y + 4 y^3)^2 + u^2 dt,   y' = u,
%   on Omega = [-1, 1]. The stationary value function is
%   V(x) = x^4 + x^2 exp(x).

p.name = 'oc1d';
p.d    = 1;
p.a    = 1;                 % Omega = [-1, 1]
p.beta = 1;                 % control weight
p.T    = 10;                % horizon of the time-dependent Galerkin solve
p.dt   = 0.2;               % output grid of the backward ODE solve

p.f   = @(X) zeros(size(X, 1), 1);                          % drift
p.q   = 1;                                                  % control direction
p.ell = @(X) 0.25 * (X.^2 .* exp(X) + 2 * X .* exp(X) + 4 * X.^3).^2;

% Gauss-Legendre points per direction (solver and indicator)
p.nq_solver    = @(NN_sel) 200;
p.nq_indicator = @(NN) 200;

% exact solution, for validation
p.V     = @(x) x.^4 + x.^2 .* exp(x);
p.gradV = @(x) 4 * x.^3 + 2 * x .* exp(x) + x.^2 .* exp(x);
end

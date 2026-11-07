function p = problem_vanderpol()
%PROBLEM_VANDERPOL  Optimal control of the Van der Pol oscillator (Section 6.3).
%
%   min_u int_0^T y1^2 + y2^2 + beta |u|^2 dt
%   y1' = y2,  y2' = -y1 + y2 (1 - y1^2) + u,   on Omega = [-3, 3]^2.

p.name = 'vanderpol';
p.d    = 2;
p.a    = [3, 3];            % Omega = [-3, 3]^2
p.beta = 0.1;
p.T    = 3;
p.dt   = 1;

p.f   = @(X) [X(:, 2), -X(:, 1) + X(:, 2) .* (1 - X(:, 1).^2)];
p.q   = [0; 1];
p.ell = @(X) sum(X.^2, 2);

% Gauss-Legendre points per direction:
%   solver    - exact for the mass matrix of the active basis
%   indicator - fixed rule, based on the largest degree in the dictionary
p.nq_solver    = @(NN_sel) 2 * max(max(NN_sel(:)), 1);
p.nq_indicator = @(NN) 2 * max(NN(:)) + 6;
end

function c = galerkin_hjb(NN_sel, p)
%GALERKIN_HJB  Time-dependent Galerkin scheme for the HJB equation (Section 3.2.1).
%
%   c = GALERKIN_HJB(NN_sel, p) projects
%       -V_t = ell + grad V . f - (1/(4 beta)) (q' grad V)^2,   V(T, .) = 0,
%   onto span{phi_j : j in NN_sel}, integrates the resulting ODE system
%   backward from t = T to t = 0 with ode15s and returns the coefficients
%   of V(0, .) in the basis of TENSOR_LEGENDRE_BASIS.
%
%   The active basis is orthonormalised with respect to the quadrature
%   weights (weighted QR) before integrating, which keeps the mass matrix
%   well conditioned as the basis grows.
%
%   The backward integration of the ODE system (9)-(10) uses p.T and p.dt.

n  = size(NN_sel, 1);
nq = p.nq_solver(NN_sel);
[Xq, wq] = gauss_legendre(nq, p.a .* ones(1, p.d));
Nq = numel(wq);

[Phi, G] = tensor_legendre_basis(Xq, NN_sel, p.a);

% W-orthonormal basis Q = Phi / R
sw = sqrt(wq);
[Qw, R] = qr(Phi .* sw, 0);
Q = Qw ./ sw;
M = Q' * (wq .* Q);                          % identity up to rounding

% directional derivatives of the basis along f and along q
fq = p.f(Xq);
Df = zeros(Nq, n);
Dq = zeros(Nq, n);
for k = 1:p.d
    Gk = reshape(G(:, k, :), Nq, n);
    Df = Df + fq(:, k) .* Gk;
    Dq = Dq + p.q(k) * Gk;
end

K   = R' \ (Phi' * (wq .* Df)) / R;          % <phi_i, f . grad phi_j>
L   = R' \ (Phi' * (wq .* p.ell(Xq)));        % <phi_i, ell>
DqR = Dq / R;                                 % q . grad of the Q basis

rhs = @(t, cq) -(M \ (K * cq - (1 / (4 * p.beta)) * (Q' * (wq .* (DqR * cq).^2)) + L));

[t_out, C] = ode15s(rhs, p.T:-p.dt:0, zeros(n, 1));
if abs(t_out(end)) > 1e-12
    warning('galerkin_hjb:incomplete', ...
        'ode15s stopped at t = %g before reaching t = 0.', t_out(end));
end

c = R \ C(end, :)';                           % back to the original basis
end

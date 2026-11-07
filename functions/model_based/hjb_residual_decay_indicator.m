function e = hjb_residual_decay_indicator(NN, sel, cand, c, p)
%HJB_RESIDUAL_DECAY_INDICATOR  Stationary HJB residual-decay indicator (24).
%
%   e = HJB_RESIDUAL_DECAY_INDICATOR(NN, sel, cand, c, p) returns, for every
%   candidate j in cand, the indicator of Definition 2,
%       e_j = | <R(V), a_k . grad phi_j> | / || a_k . grad phi_j ||,
%   where V = sum_{i in sel} c_i phi_i is the current approximation,
%       R(V) = -(1/(4 beta)) (q' grad V)^2 + grad V . f + ell     (22)
%       a_k  = f - (1/(2 beta)) q (q' grad V)                      (23)
%   and the L^2(Omega) inner products are computed by Gauss-Legendre
%   quadrature with p.nq_indicator(NN) points per direction.
%   e_j = 0 when the denominator vanishes (e.g. for the constant function).

nq = p.nq_indicator(NN);
[Xq, wq] = gauss_legendre(nq, p.a .* ones(1, p.d));
Nq = numel(wq);

[~, gradV] = eval_expansion(Xq, NN(sel, :), c, p.a);
fq = p.f(Xq);
qg = gradV * p.q(:);                                   % q' grad V

Rv = p.ell(Xq) + sum(gradV .* fq, 2) - qg.^2 / (4 * p.beta);
A  = fq - (qg / (2 * p.beta)) * p.q(:)';               % a_k(x), Nq-by-d

[~, Gc] = tensor_legendre_basis(Xq, NN(cand, :), p.a);
Lphi = reshape(sum(A .* Gc, 2), Nq, numel(cand));      % a_k . grad phi_j

num = abs(wq' * (Rv .* Lphi));
den = sqrt(wq' * Lphi.^2);
e = zeros(numel(cand), 1);
ok = den > 0;
e(ok) = num(ok) ./ den(ok);
end

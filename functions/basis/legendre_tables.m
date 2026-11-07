function [L, dL] = legendre_tables(X, max_deg)
%LEGENDRE_TABLES  Orthonormal Legendre polynomials and their derivatives.
%
%   [L, dL] = LEGENDRE_TABLES(X, max_deg) evaluates
%       p_n(x) = sqrt((2n+1)/2) * P_n(x),   n = 0, ..., max_deg,
%   which are orthonormal on [-1, 1], at every entry of the N-by-d array X.
%   L(:, :, n+1) = p_n(X) and dL(:, :, n+1) = p_n'(X).
%
%   The three-term recurrence
%       (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1},
%       P_{n+1}'      = P_{n-1}' + (2n+1) P_n
%   is used, so any degree is supported.

[N, d] = size(X);
L  = zeros(N, d, max_deg + 1);
dL = zeros(N, d, max_deg + 1);

P_prev  = ones(N, d);   dP_prev = zeros(N, d);       % P_0, P_0'
L(:, :, 1) = sqrt(1/2) * P_prev;
if max_deg == 0
    return
end

P  = X;                 dP = ones(N, d);             % P_1, P_1'
L(:, :, 2)  = sqrt(3/2) * P;
dL(:, :, 2) = sqrt(3/2) * dP;

for n = 1:max_deg-1
    P_next  = ((2*n + 1) * X .* P - n * P_prev) / (n + 1);
    dP_next = dP_prev + (2*n + 1) * P;
    L(:, :, n+2)  = sqrt((2*n + 3) / 2) * P_next;
    dL(:, :, n+2) = sqrt((2*n + 3) / 2) * dP_next;
    P_prev = P;   P = P_next;
    dP_prev = dP; dP = dP_next;
end
end

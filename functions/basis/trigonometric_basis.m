function [Phi, dPhi] = trigonometric_basis(s, m)
%TRIGONOMETRIC_BASIS  Trigonometric dictionary of Section 6.1 and its derivative.
%
%   [Phi, dPhi] = TRIGONOMETRIC_BASIS(s, m) evaluates, at the points s (column),
%       [1/sqrt(2 pi), cos(s)/sqrt(pi), sin(s)/sqrt(pi), ...,
%        cos(m s)/sqrt(pi), sin(m s)/sqrt(pi)],
%   orthonormal in L^2(-pi, pi). Column 2k is cos(k s), column 2k+1 is sin(k s).
Phi  = zeros(numel(s), 2*m + 1);
dPhi = zeros(numel(s), 2*m + 1);
Phi(:, 1) = 1 / sqrt(2 * pi);
for k = 1:m
    Phi(:, 2*k)    =  cos(k * s) / sqrt(pi);
    Phi(:, 2*k+1)  =  sin(k * s) / sqrt(pi);
    dPhi(:, 2*k)   = -k * sin(k * s) / sqrt(pi);
    dPhi(:, 2*k+1) =  k * cos(k * s) / sqrt(pi);
end
end

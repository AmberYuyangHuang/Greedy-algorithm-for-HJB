function [err_L2, err_H1] = validation_errors(V, V_true, gradV, gradV_true)
%VALIDATION_ERRORS  Relative L^2 and H^1 validation errors (25).
%
%   err_L2 = ||V - V_true||_2 / ||V_true||_2
%   err_H1 = ||gradV - gradV_true||_F / ||gradV_true||_F
%
%   V, V_true are N-by-1 and gradV, gradV_true are N-by-d.

err_L2 = norm(V - V_true) / norm(V_true);
err_H1 = norm(gradV - gradV_true, 'fro') / norm(gradV_true, 'fro');
end

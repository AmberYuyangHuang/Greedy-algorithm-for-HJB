%% Table 1 - one-dimensional benchmark for basis selection (Section 6.1)
%
%   gamma V(x) = r(x) + b(x) V'(x),  x in [-pi, pi],            eq. (29)
%   gamma = 0.1,  b(x) = 0.05 sin(x),  exact solution V(x) = cos(x) + cos(10 x),
%   dictionary of P = 101 trigonometric functions (cos and sin up to frequency 50).
%
% For each initial set Gamma^(1) in {2}, {20}, {1} the script runs
%   - Algorithm 1 (model-based) with the residual-decay indicator e_PDE (18),
%     coefficients from the Galerkin system (gamma M - B) c = r, and
%   - Algorithm 2 (data-driven) with the supervised-loss indicator e_SL (26),
%     coefficients by least squares on N = 60 equispaced samples of V and V'
%     (no data selection),
% and prints the selected indices and the relative L^2 validation errors (25).
% It also prints the model-based initial choice j* = argmax_j e_PDE^(0).
 
clear; close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'functions')));

gamma = 0.1;
b  = @(x) 0.05 * sin(x);
V  = @(x) cos(x) + cos(10 * x);
dV = @(x) -sin(x) - 10 * sin(10 * x);
r  = @(x) gamma * V(x) - b(x) .* dV(x);

m = 50;                                   % P = 2m + 1 = 101
P = 2 * m + 1;

% L^2(-pi, pi) inner products: the trapezoidal rule on a periodic grid is
% exact for the trigonometric polynomials that appear here
nq = 1024;
xq = -pi + 2 * pi * (0:nq-1)' / nq;
wq = (2 * pi / nq) * ones(nq, 1);
[Phi_q, dPhi_q] = trigonometric_basis(xq, m);
A_q = gamma * Phi_q - b(xq) .* dPhi_q;    % A phi_j = gamma phi_j - b phi_j'
r_q = r(xq);

% Galerkin matrices on the whole dictionary: (gamma M - B) c = r on Gamma
M  = Phi_q' * (wq .* Phi_q);
B  = Phi_q' * (wq .* (b(xq) .* dPhi_q));
rv = Phi_q' * (wq .* r_q);

% training samples (data-driven setting) and validation grid
N = 60;
x_train = linspace(-pi, pi, N)';
h = 2 * pi / (N - 1);
x_valid = linspace(-pi + h/2, pi - h/2, 100)';     % training grid shifted by h/2
[Phi_t, dPhi_t] = trigonometric_basis(x_train, m);
X_train = [Phi_t; dPhi_t];                         % gradient-augmented features
b_train = [V(x_train); dV(x_train)];

solve.PDE = @(G) (gamma * M(G, G) - B(G, G)) \ rv(G);    % Galerkin projection
solve.SL  = @(G) X_train(:, G) \ b_train;                % least squares
Phi_v = trigonometric_basis(x_valid, m);
V_valid = V(x_valid);

% ---------------------------------------------------------------- initialisation
% V^(0) = 0, so R(V^(0)) = -r and e_PDE^(0) = |<r, A phi_j>| / ||A phi_j||
e0 = abs(wq' * (r_q .* A_q)) ./ sqrt(wq' * A_q.^2);
[~, j0] = max(e0);
fprintf('Model-based initialisation: j* = argmax_j e_PDE^(0) = %d\n\n', j0);

% ---------------------------------------------------------------- selection paths
inits = [2, 20, 1];
indicators = {'PDE', 'SL'};
max_iter = 2;
tol = 1e-12;                              % stop once V is recovered

fprintf('%-8s %-10s %-8s %-8s %-12s %-12s\n', 'Gamma1', 'indicator', ...
        'j*(k=1)', 'j*(k=2)', 'err(k=1)', 'err(k=2)');
table1 = struct('init', {}, 'indicator', {}, 'selected', {}, 'err_L2', {});
for init = inits
    for t = 1:numel(indicators)
        G = init;
        c = solve.(indicators{t})(G);
        selected = [];
        err = [];
        for k = 1:max_iter
            cand = setdiff(1:P, G);
            switch indicators{t}
                case 'PDE'      % residual-decay indicator (18), L_k = A
                    res = A_q(:, G) * c - r_q;
                    e = abs(wq' * (res .* A_q(:, cand))) ./ sqrt(wq' * A_q(:, cand).^2);
                case 'SL'       % supervised-loss indicator (26), omega_1 = omega_2 = 1
                    res = b_train - X_train(:, G) * c;
                    e = abs(X_train(:, cand)' * res) ./ sqrt(sum(X_train(:, cand).^2, 1))';
            end
            [~, i] = max(e);
            G = [G, cand(i)]; %#ok<AGROW>
            c = solve.(indicators{t})(G);
            selected(k) = cand(i); %#ok<SAGROW>
            err(k) = norm(Phi_v(:, G) * c - V_valid) / norm(V_valid); %#ok<SAGROW>
            if err(k) < tol, break; end
        end
        table1(end+1) = struct('init', init, 'indicator', indicators{t}, ...
                               'selected', selected, 'err_L2', err); %#ok<SAGROW>
        sel_str = arrayfun(@num2str, selected, 'UniformOutput', false);
        err_str = arrayfun(@(x) sprintf('%.2e', x), err, 'UniformOutput', false);
        sel_str(end+1:max_iter) = {'-'};
        err_str(end+1:max_iter) = {'-'};
        fprintf('%-8s %-10s %-8s %-8s %-12s %-12s\n', sprintf('{%d}', init), ...
                indicators{t}, sel_str{:}, err_str{:});
    end
end

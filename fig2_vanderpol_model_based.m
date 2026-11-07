%% Figure 2 - Van der Pol oscillator, model-based setting (Section 6.3)
%
% Algorithm 1 with the time-dependent Galerkin scheme of Section 3.2.1
% (T = 3, beta = 0.1) on two Legendre dictionaries on [-3, 3]^2, starting
% from the constant basis function and running until the dictionary is
% exhausted (k_max = P):
%   hyperbolic cross, J = 4 (P = 52)  and  total degree, M = 8 (P = 45).
%   (a) random basis selection, mean over 100 runs,
%   (b) greedy selection with the residual-decay indicator e_HJB (24).
% Validation on the 1000 points of data/vanderpol2d_valid.mat.

clear; close all;
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'functions')));

n_runs_random = 100;
seed = 2026;

p = problem_vanderpol();
D = load_dataset(fullfile(root, 'data', 'vanderpol2d_train.mat'), ...
                 fullfile(root, 'data', 'vanderpol2d_valid.mat'), 2, p.a);
valid = struct('X', D.Xv, 'V', D.zv, 'gradV', D.gv);

dicts = struct('name', {'hc', 'td'}, ...
               'NN', {hyperbolic_cross_set(2, 4), total_degree_set(2, 8)});
for k = 1:numel(dicts)
    fprintf('Figure 2(b): greedy selection, %s basis\n', dicts(k).name);
    res.greedy.(dicts(k).name) = model_based_greedy( ...
        p, dicts(k).NN, 'greedy', valid, 1, seed);
    fprintf('Figure 2(a): random selection, %s basis\n', dicts(k).name);
    res.random.(dicts(k).name) = model_based_greedy( ...
        p, dicts(k).NN, 'random', valid, n_runs_random, seed);
end

panels = {'2d_galerkin_random', 'random';     % (a)
          '2d_galerkin_Ln',     'greedy'};    % (b)
for k = 1:size(panels, 1)
    R = res.(panels{k, 2});
    hc = struct('x', 1 + (1:size(R.hc.L2, 2)), 'L2', mean(R.hc.L2, 1), 'H1', mean(R.hc.H1, 1));
    td = struct('x', 1 + (1:size(R.td.L2, 2)), 'L2', mean(R.td.L2, 1), 'H1', mean(R.td.H1, 1));
    fig = plot_model_based_2d(hc, td);
    save_figure(fig, fullfile(root, 'figures', [panels{k, 1} '.png']));
end

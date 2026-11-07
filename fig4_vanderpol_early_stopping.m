%% Figure 4 - Van der Pol oscillator, data-driven setting with early stopping (Section 6.3)
%
% Figure 3(a) (basis selection e_SL (26) + data selection, score (28),
% Lasso lambda = 1e-4) with the stopping thresholds of Algorithm 2: no
% further basis functions are added once max_j e_j < eps_basis = 0.05, and
% no further samples once max_i score_i < eps_data = 0.1. The vertical
% lines mark the mean stopping points over 100 runs.
 
clear; close all;
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'functions')));

n_runs = 100;
seed = 2026;                % same seed as Figure 3
eps_basis = 0.05;
eps_data = 0.1;

D = load_dataset(fullfile(root, 'data', 'vanderpol2d_train.mat'), ...
                 fullfile(root, 'data', 'vanderpol2d_valid.mat'), 2, [3, 3]);
NN = hyperbolic_cross_set(2, 4);

opt = struct('basis_rule', 'SL', 'data_rule', 'score', ...
             'n_iter', 51, 'n_init_data', 1, 'n_init_basis', 1, ...
             'lambda', 1e-4, 'omega_basis', [1, 1], 'omega_data', [1, 1], ...
             'eps_basis', eps_basis, 'eps_data', eps_data, ...
             'n_runs', n_runs, 'seed', seed);
res = data_driven_greedy(D, NN, opt);

fprintf('Figure 4: mean stop at %.2f basis functions and %.2f data points (%d runs)\n', ...
    mean(res.basis_stop, 'omitnan'), mean(res.data_stop, 'omitnan'), size(res.L2, 1));

style = struct('legend_loc', 'northeast', 'y_lim', [3e-3, 3], ...
               'tick_step', 5, 'show_stop', true);
fig = plot_data_driven(res, style);
save_figure(fig, fullfile(root, 'figures', '2d_stop.png'));

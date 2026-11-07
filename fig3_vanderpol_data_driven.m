%% Figure 3 - Van der Pol oscillator, data-driven setting (Section 6.3)
%
% Algorithm 2 on the hyperbolic cross dictionary J = 4 (P = 52) with
% Lasso (lambda = 1e-4). Each run starts from one random training sample
% and the constant basis function and adds one sample and one basis
% function per iteration (51 iterations, k_max = P):
%   (a) basis selection, e_SL (26)  + data selection, score (28)
%   (b) random basis                + data selection
%   (c) basis selection             + random data
%   (d) random basis                + random data

clear; close all;
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'functions')));

n_runs = 100;
seed = 2026;

panels = { % panel  basis     data      legend       file
           'a',     'SL',     'score',  'northeast', '2d_residue_data';
           'b',     'random', 'score',  'southwest', '2d_random_data';
           'c',     'SL',     'random', 'northeast', '2d_residue_random';
           'd',     'random', 'random', 'southwest', '2d_random_random'};

D = load_dataset(fullfile(root, 'data', 'vanderpol2d_train.mat'), ...
                 fullfile(root, 'data', 'vanderpol2d_valid.mat'), 2, [3, 3]);
NN = hyperbolic_cross_set(2, 4);

opt = struct('n_iter', 51, 'n_init_data', 1, 'n_init_basis', 1, ...
             'lambda', 1e-4, 'omega_basis', [1, 1], 'omega_data', [1, 1], ...
             'n_runs', n_runs, 'seed', seed);
for k = 1:size(panels, 1)
    fprintf('Figure 3(%s): %s basis, %s data\n', panels{k, 1:3});
    opt.basis_rule = panels{k, 2};
    opt.data_rule  = panels{k, 3};
    res.(panels{k, 1}) = data_driven_greedy(D, NN, opt);
end

for k = 1:size(panels, 1)
    style = struct('legend_loc', panels{k, 4}, 'y_lim', [3e-3, 3], ...
                   'tick_step', 5, 'show_stop', false);
    fig = plot_data_driven(res.(panels{k, 1}), style);
    save_figure(fig, fullfile(root, 'figures', [panels{k, 5} '.png']));
end

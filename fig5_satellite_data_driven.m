%% Figure 5 - six-state rigid body model of a satellite, data-driven setting (Section 6.4)
%
% Algorithm 2 on the hyperbolic cross dictionary J = 4 in d = 6
% (P = 537) with Lasso (lambda = 1e-4), on the domain
% [-pi/3, pi/3]^3 x [-pi/4, pi/4]^3:
%   (a) basis selection, e_SL (26) + data selection, score (28)
%                                     10 samples + constant basis, 101 iterations
%   (b) random basis + data selection 99 samples + 199 basis functions, 337 iterations
%   (c) basis selection + random data as (a)
%   (d) random basis + random data    as (b)
% Validation on the 1000 points of data/satellite6d_valid.mat.
 

clear; close all;
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'functions')));

n_runs = 100;
seed = 2026;

panels = { % panel  basis     data      n_iter  init data  init basis  file
           'a',     'SL',     'score',  101,    10,        1,          '6d_residue_data';
           'b',     'random', 'score',  337,    99,        199,        '6d_random_data';
           'c',     'SL',     'random', 101,    10,        1,          '6d_residue_random';
           'd',     'random', 'random', 337,    99,        199,        '6d_random_random'};

a = [pi/3, pi/3, pi/3, pi/4, pi/4, pi/4];
D = load_dataset(fullfile(root, 'data', 'satellite6d_train.mat'), ...
                 fullfile(root, 'data', 'satellite6d_valid.mat'), 6, a);
NN = hyperbolic_cross_set(6, 4);

for k = 1:size(panels, 1)
    fprintf('Figure 5(%s): %s basis, %s data\n', panels{k, 1:3});
    opt = struct('basis_rule', panels{k, 2}, 'data_rule', panels{k, 3}, ...
                 'n_iter', panels{k, 4}, 'n_init_data', panels{k, 5}, ...
                 'n_init_basis', panels{k, 6}, 'lambda', 1e-4, ...
                 'omega_basis', [1, 1], 'omega_data', [1, 1], ...
                 'n_runs', n_runs, 'seed', seed);
    res.(panels{k, 1}) = data_driven_greedy(D, NN, opt);
end

for k = 1:size(panels, 1)
    style = struct('legend_loc', 'northeast', 'y_lim', [1e-2, 2], ...
                   'tick_step', 20, 'show_stop', false);
    fig = plot_data_driven(res.(panels{k, 1}), style);
    save_figure(fig, fullfile(root, 'figures', [panels{k, 7} '.png']));
end

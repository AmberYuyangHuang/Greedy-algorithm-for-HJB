%% Figure 1 - one-dimensional optimal control problem, model-based (Section 6.2)
%
% Algorithm 1 with the time-dependent Galerkin scheme of Section 3.2.1
% (T = 10) on the Legendre basis of degree at most 10 (P = 11), starting
% from the constant basis function and running until the dictionary is
% exhausted (k_max = P):
%   (a) random basis selection, mean over n_runs_random runs,
%   (b) greedy selection with the residual-decay indicator e_HJB (30).
% Validation on 1000 equispaced points of [-1, 1] against the exact
% stationary value function V(x) = x^4 + x^2 exp(x).
 
clear; close all;
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'functions')));

n_runs_random = 100;
seed = 1;

p = problem_oc1d();
NN = total_degree_set(1, 10);
valid.X = linspace(-1, 1, 1000)';
valid.V = p.V(valid.X);
valid.gradV = p.gradV(valid.X);

fprintf('Figure 1(b): greedy selection\n');
res.greedy = model_based_greedy(p, NN, 'greedy', valid, 1, seed);
fprintf('Figure 1(a): random selection\n');
res.random = model_based_greedy(p, NN, 'random', valid, n_runs_random, seed);

label = 'total degree basis, $M=10$';
x = 1 + (1:size(res.greedy.L2, 2));
fig = plot_model_based_1d(x, mean(res.random.L2, 1), mean(res.random.H1, 1), label);
save_figure(fig, fullfile(root, 'figures', '1d_random.png'));
fig = plot_model_based_1d(x, res.greedy.L2, res.greedy.H1, label);
save_figure(fig, fullfile(root, 'figures', '1d_pde.png'));

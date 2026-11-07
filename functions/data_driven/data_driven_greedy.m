function out = data_driven_greedy(D, NN, opt)
%DATA_DRIVEN_GREEDY  Algorithm 2 (data-driven greedy algorithm, Section 5.3).
%
%   out = DATA_DRIVEN_GREEDY(D, NN, opt) runs opt.n_runs independent
%   repetitions of the joint basis and data selection loop. Each iteration
%     1. adds one training sample (data rule) and refits,
%     2. adds one basis function (basis rule) and refits,
%     3. records the validation errors (25).
%   Coefficients are fitted by gradient-augmented regression (Section 3.3,
%   see FIT_COEFFICIENTS). The constant basis function is always active.
%
%   D    data struct: X, z, g (training pool), Xv, zv, gv (validation),
%        a (half-widths of the box Omega); see LOAD_DATASET. The basis
%        rule 'PDE' also needs f (N-by-d drift), ell (N-by-1 running cost)
%        and beta of the stationary HJB equation (2) with q = I at the
%        training samples (see GENERATE_HJB35D_DATA).
%   NN   P-by-d dictionary of multi-indices
%   opt  struct with fields (defaults in brackets)
%     basis_rule    'SL'     supervised-loss indicator e_SL (26)
%                   'PDE'    empirical residual-decay indicator e~_PDE (27)
%                   'random' uniformly random candidate
%                   'fixed'  keep the initial basis (e.g. the full basis)
%     data_rule     'score'  data score (28)
%                   'random' uniformly random sample
%     n_iter        number of iterations (k_max)
%     n_init_data   initial number of training samples                [1]
%     n_init_basis  initial number of basis functions (constant first) [1]
%     lambda        Lasso parameter (0 = least squares)            [1e-4]
%     lasso_reltol  convergence tolerance 'RelTol' of LASSO        [1e-8]
%     omega_basis   weights [omega_1, omega_2] in (26)              [1 1]
%     omega_data    weights [omega_1, omega_2] in (28)              [1 1]
%     eps_basis     threshold eps_basis of Algorithm 2                [0]
%     eps_data      threshold eps_data of Algorithm 2                 [0]
%     n_runs        number of independent runs                      [100]
%     seed          base seed                                      [2026]
%
%   Stopping: as in Algorithm 2, basis (data) selection stops for good once
%   the largest indicator (score) falls below eps_basis (eps_data). Unlike
%   Algorithm 2, the loop then continues to n_iter with the frozen iterate,
%   so that all runs have the same length; the stopping sizes are returned
%   in basis_stop and data_stop.
%
%   out fields (n_runs-by-n_iter unless stated)
%     L2, H1             validation errors (25) after each iteration
%     n_basis, n_data    active basis / data sizes after each iteration
%     basis_stop, data_stop  1-by-n_runs sizes at which basis / data
%                        selection stopped (NaN if never)
%     selected_basis, selected_data   1-by-n_runs cells of index lists
%     opt                the options used (with defaults filled in)

opt = fill_defaults(opt, struct('n_init_data', 1, 'n_init_basis', 1, ...
    'lambda', 1e-4, 'lasso_reltol', 1e-8, 'omega_basis', [1 1], 'omega_data', [1 1], ...
    'eps_basis', 0, 'eps_data', 0, 'n_runs', 100, 'seed', 2026));
if strcmp(opt.basis_rule, 'PDE') && ~all(isfield(D, {'f', 'ell', 'beta'}))
    error('basis_rule ''PDE'' needs D.f, D.ell and D.beta (HJB drift, running cost, control weight).');
end

N = size(D.X, 1);
P = size(NN, 1);
R = opt.n_runs;
T = opt.n_iter;

out.L2 = nan(R, T);       out.H1 = nan(R, T);
out.n_basis = nan(R, T);  out.n_data = nan(R, T);
out.basis_stop = nan(1, R);
out.data_stop  = nan(1, R);
out.selected_basis = cell(1, R);
out.selected_data  = cell(1, R);
out.opt = opt;

for r = 1:R
    t_run = tic;
    rng(opt.seed + r);
    data_perm  = randperm(N);
    basis_perm = 1 + randperm(P - 1);

    % initial data and basis
    sel_d = data_perm(1:opt.n_init_data);
    data_order = data_perm(opt.n_init_data+1:end);
    used = false(N, 1);
    used(sel_d) = true;

    if strcmp(opt.basis_rule, 'fixed') && opt.n_init_basis == P
        sel_b = 1:P;
        basis_order = [];
    else
        sel_b = [1, basis_perm(1:opt.n_init_basis-1)];
        basis_order = basis_perm(opt.n_init_basis:end);
    end

    c = refit(D, NN, sel_d, sel_b, opt.lambda, opt.lasso_reltol);
    data_stop = false;
    basis_stop = false;

    for it = 1:T
        % 1. data step
        switch opt.data_rule
            case 'score'
                if ~data_stop
                    pool = find(~used);
                    s = data_score(D.X(pool, :), D.z(pool), D.g(pool, :), ...
                                   NN(sel_b, :), c, D.a, opt.omega_data);
                    [s_max, i] = max(s);
                    if s_max < opt.eps_data
                        data_stop = true;
                        out.data_stop(r) = numel(sel_d);
                    else
                        sel_d(end+1) = pool(i); %#ok<AGROW>
                        used(pool(i)) = true;
                    end
                end
            case 'random'
                sel_d(end+1) = data_order(1); %#ok<AGROW>
                used(data_order(1)) = true;
                data_order(1) = [];
            otherwise
                error('Unknown data rule "%s" (use ''score'' or ''random'').', opt.data_rule);
        end
        [c, Phi, G] = refit(D, NN, sel_d, sel_b, opt.lambda, opt.lasso_reltol);

        % 2. basis step
        cand = setdiff(1:P, sel_b);
        new_b = [];
        switch opt.basis_rule
            case {'SL', 'PDE'}
                if ~basis_stop && ~isempty(cand)
                    m = numel(sel_d);
                    gradV_s = reshape(reshape(G, [], numel(sel_b)) * c, m, []);
                    [Phi_c, G_c] = tensor_legendre_basis(D.X(sel_d, :), NN(cand, :), D.a);
                    if strcmp(opt.basis_rule, 'SL')                % (26)
                        res_f = D.z(sel_d) - Phi * c;
                        res_g = D.g(sel_d, :) - gradV_s;
                        e = supervised_loss_indicator(Phi_c, G_c, res_f, res_g, opt.omega_basis);
                    else                                            % (27)
                        e = empirical_residual_decay_indicator(G_c, gradV_s, ...
                                D.f(sel_d, :), D.ell(sel_d), D.beta);
                    end
                    [e_max, i] = max(e);
                    if e_max < opt.eps_basis
                        basis_stop = true;
                        out.basis_stop(r) = numel(sel_b);
                    else
                        new_b = cand(i);
                    end
                end
            case 'random'
                if ~isempty(basis_order)
                    new_b = basis_order(1);
                    basis_order(1) = [];
                end
            case 'fixed'
            otherwise
                error('Unknown basis rule "%s" (use ''SL'', ''PDE'', ''random'' or ''fixed'').', ...
                      opt.basis_rule);
        end
        if ~isempty(new_b)
            sel_b(end+1) = new_b; %#ok<AGROW>
            c = refit(D, NN, sel_d, sel_b, opt.lambda, opt.lasso_reltol);
        end

        % 3. validation errors (25)
        [V, gradV] = eval_expansion(D.Xv, NN(sel_b, :), c, D.a);
        [out.L2(r, it), out.H1(r, it)] = validation_errors(V, D.zv, gradV, D.gv);
        out.n_basis(r, it) = numel(sel_b);
        out.n_data(r, it)  = numel(sel_d);
    end

    out.selected_basis{r} = sel_b;
    out.selected_data{r}  = sel_d;
    fprintf('  run %3d/%d: final L2 = %.3e, H1 = %.3e  (%.1f s)\n', ...
        r, R, out.L2(r, end), out.H1(r, end), toc(t_run));
end
end


function [c, Phi, G] = refit(D, NN, sel_d, sel_b, lambda, reltol)
[Phi, G] = tensor_legendre_basis(D.X(sel_d, :), NN(sel_b, :), D.a);
c = fit_coefficients(Phi, G, D.z(sel_d), D.g(sel_d, :), lambda, reltol);
end


function opt = fill_defaults(opt, defaults)
names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(opt, names{k})
        opt.(names{k}) = defaults.(names{k});
    end
end
end

function out = model_based_greedy(p, NN, rule, valid, n_runs, seed)
%MODEL_BASED_GREEDY  Algorithm 1 (model-based greedy algorithm, Section 4.5).
%
%   out = MODEL_BASED_GREEDY(p, NN, rule, valid, n_runs, seed) starts from
%   the constant basis function and adds one basis function per iteration
%   until the whole dictionary NN is used (k_max = P). After each
%   enrichment the time-dependent Galerkin scheme of Section 3.2.1 is
%   solved (GALERKIN_HJB) and the approximation is validated.
%
%   rule    'greedy' - stationary HJB residual-decay indicator e_HJB (24)
%           'random' - uniformly random candidate
%   valid   struct with fields X, V, gradV (validation points and values)
%   n_runs  number of independent runs (the greedy rule is deterministic)
%   seed    base seed; run r uses rng(seed + r)
%
%   out.L2, out.H1   n_runs-by-(P-1) validation errors (25); column k is
%                    the error with k+1 basis functions
%   out.selected     n_runs-by-P selection order

P = size(NN, 1);
n_iter = P - 1;
out.L2 = zeros(n_runs, n_iter);
out.H1 = zeros(n_runs, n_iter);
out.selected = zeros(n_runs, P);

for r = 1:n_runs
    rng(seed + r);
    order = 1 + randperm(P - 1);              % used by the random rule

    sel = 1;                                  % constant basis function
    c = galerkin_hjb(NN(sel, :), p);

    for it = 1:n_iter
        cand = setdiff(1:P, sel);
        switch rule
            case 'greedy'
                e = hjb_residual_decay_indicator(NN, sel, cand, c, p);
                [~, i] = max(e);
                new = cand(i);
            case 'random'
                new = order(it);
            otherwise
                error('Unknown rule "%s" (use ''greedy'' or ''random'').', rule);
        end

        sel = [sel, new]; %#ok<AGROW>
        c = galerkin_hjb(NN(sel, :), p);

        [V, gradV] = eval_expansion(valid.X, NN(sel, :), c, p.a);
        [out.L2(r, it), out.H1(r, it)] = ...
            validation_errors(V, valid.V, gradV, valid.gradV);
    end
    out.selected(r, :) = sel;
    fprintf('  %s run %d/%d: final L2 = %.3e, H1 = %.3e\n', ...
        rule, r, n_runs, out.L2(r, end), out.H1(r, end));
end
end

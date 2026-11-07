function fig = plot_data_driven(res, style)
%PLOT_DATA_DRIVEN  Error decay plot in the style of Figures 3-5.
%
%   fig = PLOT_DATA_DRIVEN(res, style) plots the mean validation errors over
%   runs against the number of selected basis functions (bottom axis) and
%   training samples (top axis), with one-standard-deviation bands computed
%   in log space.
 
clr_L2 = [0.2157, 0.4706, 0.7490];
clr_H1 = [0.8941, 0.1020, 0.1098];
alpha_band = 0.15;
lw = 1.8;  msz = 5;  fsz = 16;

n_iter = size(res.L2, 2);
x = res.opt.n_init_basis + (1:n_iter);   % one basis function per iteration

[L2_mu, L2_lo, L2_hi] = log_band(res.L2);
[H1_mu, H1_lo, H1_hi] = log_band(res.H1);

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 16 12]);
ax_pos = [0.14, 0.14, 0.78, 0.68];
ax1 = axes('Parent', fig, 'Position', ax_pos, 'FontSize', fsz, ...
           'FontName', 'Times New Roman', 'TickLabelInterpreter', 'latex', ...
           'YScale', 'log', 'Box', 'on', 'LineWidth', 0.8, ...
           'GridAlpha', 0.25, 'GridLineStyle', '--');
hold(ax1, 'on');
grid(ax1, 'on');

fill(ax1, [x, fliplr(x)], [L2_lo, fliplr(L2_hi)], clr_L2, 'FaceAlpha', alpha_band, ...
     'EdgeColor', 'none', 'HandleVisibility', 'off');
fill(ax1, [x, fliplr(x)], [H1_lo, fliplr(H1_hi)], clr_H1, 'FaceAlpha', alpha_band, ...
     'EdgeColor', 'none', 'HandleVisibility', 'off');

semilogy(ax1, x, L2_mu, '-o', 'Color', clr_L2, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_L2, 'DisplayName', '$L^2$ error');
semilogy(ax1, x, H1_mu, '--s', 'Color', clr_H1, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_H1, 'DisplayName', '$H^1$ error');

if style.show_stop
    basis_stop = mean(res.basis_stop, 'omitnan');
    data_stop  = mean(res.data_stop, 'omitnan');
    if ~isnan(basis_stop)
        xline(ax1, basis_stop, '--', 'Color', [0.8500, 0.3250, 0.0980], ...
              'LineWidth', 2.2, 'DisplayName', 'Basis stop');
    end
    if ~isnan(data_stop)
        xline(ax1, data_stop, '-.', 'Color', [0.4940, 0.1840, 0.5560], ...
              'LineWidth', 2.2, 'DisplayName', 'Data stop');
    end
end

x_ticks = x(1):style.tick_step:x(end);
ylim(ax1, style.y_lim);
set(ax1, 'XLim', [x(1), x(end)], 'XLimMode', 'manual', 'XTick', x_ticks, ...
         'XTickLabel', arrayfun(@num2str, x_ticks, 'UniformOutput', false));
xlabel(ax1, 'Number of selected basis functions', 'FontSize', fsz, 'Interpreter', 'latex');
ylabel(ax1, 'Validation error', 'FontSize', fsz, 'Interpreter', 'latex');
legend(ax1, 'Location', style.legend_loc, 'Interpreter', 'latex', 'FontSize', fsz, ...
       'EdgeColor', [0.8 0.8 0.8], 'Box', 'on');

% top axis: number of training samples (one sample per iteration)
ax2 = axes('Parent', fig, 'Position', ax_pos, 'XAxisLocation', 'top', ...
           'YAxisLocation', 'right', 'Color', 'none', 'XColor', 'k', ...
           'YColor', 'none', 'FontSize', fsz, 'FontName', 'Times New Roman', ...
           'TickLabelInterpreter', 'latex', 'LineWidth', 0.8, 'Box', 'off');
linkaxes([ax1, ax2], 'x');
data_ticks = x_ticks + (res.opt.n_init_data - res.opt.n_init_basis);
set(ax2, 'XLim', [x(1), x(end)], 'XLimMode', 'manual', 'XTick', x_ticks, ...
         'XTickLabel', arrayfun(@num2str, data_ticks, 'UniformOutput', false));
xlabel(ax2, 'Number of training data points', 'FontSize', fsz, 'Interpreter', 'latex');
set(get(ax2, 'XLabel'), 'Units', 'normalized', 'Position', [0.5, 1.12, 0]);
end


function [mu, lo, hi] = log_band(E)
% Mean over runs and a one-standard-deviation band of the lognormal
% distribution with the same mean and standard deviation.
mu  = mean(E, 1);
sd  = std(E, 0, 1);
cv2 = (sd ./ max(mu, 1e-16)).^2;
fac = exp(sqrt(log(1 + cv2)));
g   = mu ./ sqrt(1 + cv2);
lo  = g ./ fac;
hi  = g .* fac;
end

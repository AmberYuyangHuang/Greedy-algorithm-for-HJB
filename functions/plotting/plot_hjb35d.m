function fig = plot_hjb35d(res, variant)
%PLOT_HJB35D  Error decay plot in the style of Figure 6.
%
%   fig = PLOT_HJB35D(res, 'adaptive')  - Figure 6(a)-(d): mean errors
%       against the mean number of selected basis functions (bottom axis)
%       and training samples (top axis).
%   fig = PLOT_HJB35D(res, 'full')      - Figure 6(e): full dictionary,
%       mean errors against the number of training samples.
 

L2 = mean(res.L2, 1, 'omitnan');
H1 = mean(res.H1, 1, 'omitnan');
n_data = mean(res.n_data, 1, 'omitnan');
adaptive = strcmp(variant, 'adaptive');
if adaptive
    x = mean(res.n_basis, 1, 'omitnan');
    fig_width = 18;
else
    x = n_data;
    fig_width = 16;
end

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 fig_width 13]);
ax1 = axes('Parent', fig);
hold(ax1, 'on');
set(ax1, 'YScale', 'log', 'FontSize', 16, 'TickLabelInterpreter', 'latex', 'Box', 'on');
grid(ax1, 'on');
if ~adaptive
    ylim(ax1, [1e-2, 1]);
end

pos = get(ax1, 'Position');
pos(2) = 0.16;
pos(4) = 0.68;
set(ax1, 'Position', pos);

semilogy(ax1, x, L2, '-o', 'LineWidth', 1.8, 'MarkerSize', 5, 'DisplayName', '$L^2$ error');
semilogy(ax1, x, H1, '--s', 'LineWidth', 1.8, 'MarkerSize', 5, 'DisplayName', '$H^1$ error');

if adaptive
    xlabel(ax1, 'Number of selected basis functions', 'FontSize', 18, 'Interpreter', 'latex');
else
    set(ax1, 'XTick', []);
    xlabel(ax1, sprintf('Full basis set: $P=%d$ basis functions', res.n_basis(1, 1)), ...
           'FontSize', 18, 'Interpreter', 'latex');
end
ylabel(ax1, 'Validation error', 'FontSize', 18, 'Interpreter', 'latex');
legend(ax1, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 16);
xlim(ax1, [min(x), max(x)]);

% top axis: number of training samples
ax2 = axes('Position', pos, 'XAxisLocation', 'top', 'YAxisLocation', 'right', ...
           'Color', 'none', 'XColor', 'k', 'YColor', 'none', 'FontSize', 16, ...
           'TickLabelInterpreter', 'latex');
linkaxes([ax1, ax2], 'x');

x_ticks = x(1):10:x(end);
if adaptive
    set(ax1, 'XTick', x_ticks);
end
set(ax2, 'XTick', x_ticks);
data_ticks = interp1(x, n_data, x_ticks, 'linear', 'extrap');
set(ax2, 'XTickLabel', arrayfun(@(v) sprintf('%d', round(v)), data_ticks, ...
         'UniformOutput', false));
xlabel(ax2, 'Number of training data points', 'FontSize', 18, 'Interpreter', 'latex');
xlim(ax2, [min(x), max(x)]);
if adaptive
    ylim(ax1, [1e-2, 1.1]);
end
end

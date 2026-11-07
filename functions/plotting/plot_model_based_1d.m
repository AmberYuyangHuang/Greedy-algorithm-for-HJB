function fig = plot_model_based_1d(x, L2, H1, label)
%PLOT_MODEL_BASED_1D  Error decay plot in the style of Figure 1.
%
%   x       number of selected basis functions
%   L2, H1  mean validation errors (same length as x)
%   label   basis description used in the legend, e.g. 'total degree basis, $M=10$'

clr_L2 = [0.2157, 0.4706, 0.7490];
clr_H1 = [0.8941, 0.1020, 0.1098];
lw = 1.8;  msz = 5;  fsz = 16;

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 16 11]);
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [16 11], ...
         'PaperPosition', [0 0 16 11]);

ax = axes('Parent', fig, 'FontSize', fsz, 'FontName', 'Times New Roman', ...
          'TickLabelInterpreter', 'latex', 'YScale', 'log', 'Box', 'on', ...
          'LineWidth', 0.8, 'GridAlpha', 0.2, 'GridLineStyle', '--');
hold(ax, 'on');
grid(ax, 'on');

semilogy(ax, x, L2, '-o', 'Color', clr_L2, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_L2, 'DisplayName', ['$L^2$ error, ' label]);
semilogy(ax, x, H1, '--s', 'Color', clr_H1, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_H1, 'DisplayName', ['$H^1$ error, ' label]);

ylim(ax, [5e-7, 1e5]);
set(ax, 'XLim', [x(1), x(end)], 'XLimMode', 'manual', 'XTick', x(1):2:x(end));

xlabel(ax, 'Number of selected basis functions', 'FontSize', 20, 'Interpreter', 'latex');
ylabel(ax, 'Validation error', 'FontSize', 20, 'Interpreter', 'latex');
legend(ax, 'Location', 'southwest', 'Interpreter', 'latex', 'FontSize', fsz, ...
       'EdgeColor', [0.8 0.8 0.8], 'Box', 'on');
end

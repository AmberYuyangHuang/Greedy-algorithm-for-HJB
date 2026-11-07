function fig = plot_model_based_2d(hc, td)
%PLOT_MODEL_BASED_2D  Error decay plot in the style of Figure 2.
%
%   hc, td  structs with fields x, L2, H1 (mean errors) for the hyperbolic
%           cross (J = 4) and total degree (M = 8) dictionaries

clr_hc_L2 = [0.2157, 0.4706, 0.7490];
clr_hc_H1 = [0.6196, 0.7922, 0.8824];
clr_td_L2 = [0.8941, 0.1020, 0.1098];
clr_td_H1 = [0.9843, 0.6039, 0.6000];
lw = 1.8;  msz = 5;  fsz = 16;

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 19 13]);
ax = axes('Parent', fig, 'FontSize', fsz, 'FontName', 'Times New Roman', ...
          'TickLabelInterpreter', 'latex', 'YScale', 'log', 'Box', 'on', ...
          'LineWidth', 0.8, 'GridAlpha', 0.25, 'GridLineStyle', '--');
hold(ax, 'on');
grid(ax, 'on');

semilogy(ax, hc.x, hc.L2, '-o', 'Color', clr_hc_L2, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_hc_L2, 'DisplayName', '$L^2$ error, hyperbolic cross basis, $J=4$');
semilogy(ax, hc.x, hc.H1, '--o', 'Color', clr_hc_H1, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_hc_H1, 'DisplayName', '$H^1$ error, hyperbolic cross basis, $J=4$');
semilogy(ax, td.x, td.L2, '-s', 'Color', clr_td_L2, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_td_L2, 'DisplayName', '$L^2$ error, total degree basis, $M=8$');
semilogy(ax, td.x, td.H1, '--s', 'Color', clr_td_H1, 'LineWidth', lw, 'MarkerSize', msz, ...
    'MarkerFaceColor', clr_td_H1, 'DisplayName', '$H^1$ error, total degree basis, $M=8$');


x_max = min(hc.x(end), td.x(end));
x_ticks = 2:2:x_max;
ylim(ax, [3e-3, 2]);
set(ax, 'XLim', [2, x_max], 'XLimMode', 'manual', 'XTick', x_ticks, ...
        'XTickLabel', arrayfun(@num2str, x_ticks, 'UniformOutput', false));

xlabel(ax, 'Number of selected basis functions', 'FontSize', 20, 'Interpreter', 'latex');
ylabel(ax, 'Validation error', 'FontSize', 20, 'Interpreter', 'latex');
legend(ax, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 13, ...
       'EdgeColor', [0.8 0.8 0.8], 'Box', 'on');
end

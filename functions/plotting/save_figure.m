function save_figure(fig, file)
%SAVE_FIGURE  Export a figure.
 
folder = fileparts(file);
if ~isempty(folder) && ~exist(folder, 'dir')
    mkdir(folder);
end
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.png')
    exportgraphics(fig, file, 'Resolution', 300);
else
    exportgraphics(fig, file, 'ContentType', 'vector');
end
fprintf('  saved %s\n', file);
end

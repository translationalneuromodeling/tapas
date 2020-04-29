function [] = tapas_display_table(summaries, figname)
%% Display a figure with the summaries.
%
% Input
%       summaries       -- Table with the summaries.
%       figname         -- Optional. String with the name for the figure.
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 1;

n = n + 1;
if nargin < n
    figname = 'Table';
end

figure('name', figname);
uitable('Data', summaries{:, :}, ...
    'ColumnName', summaries.Properties.VariableNames,...
    'RowName', summaries.Properties.RowNames, ...
    'Units', 'Normalized', ...
    'Position',[0, 0, 1, 1]);

end

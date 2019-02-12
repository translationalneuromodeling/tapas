function [summaries] = tapas_sem_display_posterior(posterior)
%% Displays a summary of the posterior estimates.
%
% Input
%       posterior       -- Estimate from the model.
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2019
%

% Plot the data.
data = posterior.data;
try
    model = posterior.model.graph{1}.htheta.model;
catch err
    try
        model = posterior.ptheta;
    catch
        error('tapas:sem:display_posterior', 'Improper input')
    end
end

summaries = [];
for i = 1:numel(data)
    fig = figure('name', sprintf('Subject #%d', i));

    [edges] = tapas_sem_plot_antisaccades(data(i).y, data(i).u);
    % Edges of the plot
    dt = edges(2) - edges(1);
    samples = posterior.ps_theta(i, :);
    fits = tapas_sem_generate_fits(data(i), samples, model);
    tapas_sem_plot_fits(data(i), fits, dt)
    format_figure(fig, data(i), fits, model);
    summary = tapas_sem_generate_summary(data(i), samples, model, i);
    summaries = [summaries; summary];

end

figure('name', 'Summary statistics');
uitable('Data', summaries{:, :}, ...
    'ColumnName', summaries.Properties.VariableNames,...
    'RowName', summaries.Properties.RowNames, ...
    'Units', 'Normalized', ...
    'Position',[0, 0, 1, 1]);

end

function format_figure(fig, data, fits, model)

fig.Color = 'w';

conds = unique(data.u.tt);
nconds = numel(conds);

for i = 1:nconds
    ax = subplot(nconds, 2, (i - 1) * 2 + 1);
    title(sprintf('Pro. condition %d', i - 1));
    ylabel('# saccs.')
    xlabel('time')
    ax = subplot(nconds, 2, (i - 1) * 2 + 2);
    title(sprintf('Anti. condition %d', i - 1));
    ylabel('# saccs.')
    xlabel('time')
end

end

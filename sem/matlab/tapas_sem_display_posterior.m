function [summaries, fits] = tapas_sem_display_posterior(posterior, time)
%% Displays a summary of the posterior estimates.
%
% Input
%       posterior       -- Estimate from the model.
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 1;

n = n + 1;
if nargin < n
    time = [];
end 

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
fits = cell(numel(data), 1);

for i = 1:numel(data)
    fig = figure('name', sprintf('Subject #%d', i));

    [edges] = tapas_sem_plot_responses(data(i).y, data(i).u);
    % For normalizing the are
    dt = edges(2) - edges(1);
    % Edges of the plot
    samples = posterior.ps_theta(i, :);
    
    % If time is empty, use the default time array
    if numel(time)
        cond_fit = tapas_sem_generate_fits(data(i), samples, model, time);
    else
         cond_fit = tapas_sem_generate_fits(data(i), samples, model);  
    end

    tapas_sem_plot_fits(data(i), cond_fit, dt)
    format_figure(fig, data(i), cond_fit, model);
    summary = tapas_sem_generate_summary(data(i), samples, model, i);

    summaries = [summaries; summary];
    fits{i} = cond_fit;
    break
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

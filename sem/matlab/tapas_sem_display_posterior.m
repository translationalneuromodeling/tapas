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

% Display the constraints matrix.
jm = model.jm;
model_type = model.name; 

if numel(regexp(model_type, 'seria'))
    model_type = 'seria';
elseif numel(regexp(model_type, 'prosa'))
    model_type = 'prosa';
else
    error('tapas:sem:display_posterior', 'Unknown model.');
end

tapas_sem_plot_constraints(jm, model_type);

% Check if the summaries are present
if ~isfield(posterior, 'summary')
    summary = tapas_sem_posterior_summary(posterior);
else
    summary = posterior.summary;
end

% Display the summary
tapas_sem_display_posterior_summary(posterior, summary);

for i = 1:numel(data)
    fig = figure('name', sprintf('Subject #%d', i));

    [edges] = tapas_sem_plot_responses(data(i).y, data(i).u);
    % For normalizing the are
    dt = edges(2) - edges(1);
    % Edges of the plot
    fit = summary.fits{i}; 
    fit = tapas_sem_normalized_fits(data(i), fit);
    tapas_sem_plot_fits(fit, dt);
    format_figure(fig, data(i), fit, model);

end

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

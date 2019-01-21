function [edges] = tapas_sem_plot_antisaccades(y, u)
%% Plot the data from a subject
%
% Input
%       y   -- Structrue with fields t and a, where t is a vector 
%               of reaction times and a the action (either zero or
%               or 1. 
%       u   -- Structure with fields tt. tt is a vector of integers
%               indicating the conditions.
%       fig -- figure handle, optional. 
% Output
%       ax  -- Handle of the figure.

% aponteeduardo@gmail.com
% copyright (C) 2019
%

nbins = 30;

% Get number of trials
ntrials = numel(u.tt);

% Gain
maxt = max(y.t);
mint = min(y.t);

% Bins
edges = linspace(mint, maxt + 0.1 * (maxt - mint), nbins);

% Get unique conditions
conds = unique(u.tt);
nconds = numel(conds);

mcounts = -inf;

axes = {};
for i = 1:nconds
    ax = subplot(nconds, 2, (i - 1) * 2 + 1);
    hold on;
    tt = y.t((conds(i) == u.tt) & (y.a == 0));
    h = histogram(tt, 'binedges', edges, 'facecolor', 'r', ...
        'edgecolor', 'none');
    mcounts = max(max(h.Values), mcounts);
    axes{end + 1} = ax;
    ax = subplot(nconds, 2, (i - 1) * 2 + 2);
    hold on;
    tt = y.t((conds(i) == u.tt) & (y.a == 1));
    h = histogram(tt, 'binedges', edges, 'facecolor', 'b', ...
        'edgecolor', 'none');
    axes{end + 1} = ax;
    mcounts = max(max(h.Values), mcounts);
end

for i = 1:nconds * 2
    ylim(axes{i}, [0, mcounts * 1.1])
end


end

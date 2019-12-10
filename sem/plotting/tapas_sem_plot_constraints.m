function [fig] = tapas_sem_plot_constraints(constraints, model)
%% Plot the constraints matrix
%
% Input
%       constraints     -- Constraints matrix
%       model           -- String with model (seria or prosa). 
%
% Output
%       fig             -- Handle of the figure created.
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

switch model
case 'seria'
    nparms = 11;
case 'prosa'
    nparms = 9;
otherwise
    error('tapas:sem:plot_constraints', 'Unknown model');
end

[nr, nc] = size(constraints);

fig = figure('name', sprintf('Constraints matrix model %s', model));
fig.Color = 'w';

colormap('gray')
imagesc([0.5, 0.5 + nc - 1], [0.5, 0.5 + nr - 1], 1.0 - constraints)
hold on

% Number of conditions
nconds = nr/nparms;

labels = cell(nc, 1);
for i = 1:nconds
    plot([0, nc], i * [nparms, nparms], 'r-')
    plot([0, nc], (i - 1) * nparms + [2, 2], ':', 'color', [0.2, 0.2, 0.2])
    plot([0, nc], (i - 1) * nparms + [4, 4], ':', 'color', [0.2, 0.2, 0.2])
    plot([0, nc], (i - 1) * nparms + [6, 6], ':', 'color', [0.2, 0.2, 0.2])
    labels{i} = sprintf('Cond. %d', i);
end

for i = 1:nc
    plot([i, i], [0, nr], ':k')
end

yticks(ceil(nparms/2):nparms:nr);
yticklabels(labels)

xlabel('free parameter')
title('Constraint matrix')

% Make tick on the right
yyaxis right
locs = [];
labels = {};

% Make nice labels to the model parameters.
switch model
case 'seria'
    for i = 1:nconds
        % Plot the missing parameter
        plot([0, nc], (i - 1) * nparms + [8, 8], ':', 'color', [0.2, 0.2, 0.2])
        locs = [locs, ...
            ((i - 1) * nparms + 1), ... Early
            ((i - 1) * nparms + 3), ... Incongruent
            ((i - 1) * nparms + 5), ... Incongruent
            ((i - 1) * nparms + 7), ... Incongruent
            ((i - 1) * nparms + 9.5)]; ... Ancilliary
        labels =  [labels {'early', 'incong.', 'inhib.', 'cong.', ...
            'ancillary'}];
    end
case 'prosa'
    for i = 1:nconds
        locs = [locs, ...
            ((i - 1) * nparms + 1), ... Early
            ((i - 1) * nparms + 3), ... Incongruent
            ((i - 1) * nparms + 5), ... Incongruent
            ((i - 1) * nparms + 7.5)]; ... Ancilliary
        labels =  [labels {'early', 'incong.', 'inhib.', 'ancillary'}];
    end
end

set(gca, 'ydir', 'reverse')
set(gca, 'ycolor', 'k')
yticks(locs)
ylim([0, nconds * nparms])
yticklabels(labels)


end

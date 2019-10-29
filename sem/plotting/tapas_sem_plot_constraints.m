function [] = tapas_sem_plot_constraints(constraints, model)
%% Plot the constraints matrix
%
% Input
%       constraints     -- Constraints matrix
%       model           -- String with model (seria or prosa). 
%                       -- Default 'seria'
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 1;

n = n + 1;
if nargin < n
    model = 'seria';
end

switch model
case 'seria'
    nparms = 11;
case 'prosa'
    nparams = 9;
otherwise
    error('tapas:sem:plot_constraints', 'Unknown model');
end

[nr, nc] = size(constraints);


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
    plot([0, nc], (i - 1) * nparms + [8, 8], ':', 'color', [0.2, 0.2, 0.2])
    labels{i} = sprintf('Cond. %d', i);
end

for i = 1:nc
    plot([i, i], [0, nr], ':k')
end

yticks(6:11:50)
yticklabels(labels)

xlabel('free parameter')
title('Constraint matrix')

% Make tick on the right
yyaxis right
locs = [];
labels = {};

for i = 1:nconds
    locs = [locs, ...
        ((i - 1) * nparms + 1), ... Early
        ((i - 1) * nparms + 3), ... Incongruent
        ((i - 1) * nparms + 5), ... Incongruent
        ((i - 1) * nparms + 7), ... Incongruent
        ((i - 1) * nparms + 9.5)]; ... Ancilliary
        labels =  [labels {'early', 'incong.', 'inhib.', 'cong.', ...
                    'ancillary'}];
end
set(gca, 'ydir', 'reverse')
set(gca, 'ycolor', 'k')
yticks(locs)
ylim([0, nconds * nparms])
yticklabels(labels)


end

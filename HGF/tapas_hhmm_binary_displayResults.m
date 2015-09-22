function tapas_hhmm_binary_displayResults(r)
% Displays results estimated by fitModel for the hmm perceptual model in the binary case
% Usage:  est = fitModel(responses, inputs); hmm_binary_displayResults(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Get tree
N = r.p_prc.N;

% Print transition matrices
for id = 1:length(N)
    disp(' ')
    disp(['Node no. ' num2str(id) ':'])
    disp(' ')
    if ~isempty(N{id}.parent)
        disp(['    Parent: ' num2str(N{id}.parent)])
    else
        disp('    Root')
    end
    if ~isempty(N{id}.children)
        disp(['    Children: ' num2str(N{id}.children)])
    else
        disp('    Production node')
    end
    if ~isempty(N{id}.V)
        disp(' ')
        disp(['    Probability of entering from parent: ' num2str(N{id}.V)])
    end
    if ~isempty(N{id}.A)
        disp(' ')
        disp('    Transition probability matrix between children: ')
        disp(' ')
        disp(N{id}.A)
    end
    if ~isempty(N{id}.B)
        disp(' ')
        disp(['    Outcome contingencies: ' num2str(N{id}.B)])
    end
end

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.7*scrsz(4),0.8*scrsz(3),0.6*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name','RW binary fit results');

% Inputs
u = r.u(:,1)-1;

% Responses
y = r.y;

% Number of trials
t = length(u);

% Subplots

% Higher level
hightraj = sum(r.traj.alpr(:,[3,4]),2);

subplot(2,1,1);
plot(1:t, hightraj, 'b', 'LineWidth', 2);
hold all;
title(['Posterior probability \alpha'' (blue) of regime 1'], 'FontWeight', 'bold');
ylabel('\alpha''');
axis([1 t -0.1 1.1]);
plot(1:t, 0.5, 'k');
xlabel('Trial number');
hold off;

% Lower level
lowtraj = sum(r.traj.alpr(:,[1,3]),2);

subplot(2,1,2);
plot(1:t, lowtraj, 'r', 'LineWidth', 2);
hold all;
plot(1:t, u, '.', 'Color', [0 0.6 0]); % inputs
if ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    if ~isempty(find(strcmp(fieldnames(r),'irr')))
        y(r.irr) = NaN; % weed out irregular responses
    end
    plot(1:t, y, '.', 'Color', [1 0.7 0]); % responses
    title(['Response y (orange), input u (green), and posterior probability \alpha'' (red) of outcome 1'], 'FontWeight', 'bold');
    ylabel('y, u, \alpha''');
    axis([1 t -0.15 1.15]);
else
    title(['Input u (green) and posterior probability \alpha'' (red) of option 1'], 'FontWeight', 'bold');
    ylabel('u, \alpha''');
    axis([1 t -0.1 1.1]);
end
plot(1:t, 0.5, 'k');
xlabel('Trial number');
hold off;

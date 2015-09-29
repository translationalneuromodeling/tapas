function tapas_rw_binary_dual_plotTraj(r)
% Plots the estimated or generated trajectories for the binary HGF perceptual model for multi-armed
% bandit situations.
%
% Usage example:  est = tapas_fitModel(responses, inputs); tapas_rw_binary_dual_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Optional plotting of responses (true or false)
ploty = true;

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.7*scrsz(4),0.8*scrsz(3),0.3*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name', 'HGF trajectories');

% Set up colors
colors = [1 0 0; 0.67 0 1; 0 0.67 1; 0.67 1 0];

% Number of bandits
b = 2;

% Number of trials
n = size(r.u,1);

% Time axis
t = ones(1,n);
ts = cumsum(t);
ts = [0, ts];

% Plot
for j=1:b
    plot(ts, [r.p_prc.v_0(j); r.traj.v(:,j)], 'Color', colors(j,:), 'LineWidth', 2);
    hold all;
    plot(0, r.p_prc.v_0(j), 'o', 'Color', colors(j,:), 'LineWidth', 2); % prior
end
plot(ts(2:end), r.u(:,1), '.', 'Color', [0 0 0]); % inputs
if (ploty == true) && ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    y = r.y(:,1);
    if ~isempty(find(strcmp(fieldnames(r),'irr')))
        y(r.irr) = NaN; % weed out irregular responses
        plot(ts(r.irr),  1.08.*ones([1 length(r.irr)]), 'x', 'Color', [1 0.7 0], 'Markersize', 11, 'LineWidth', 2); % irregular responses
    end
    for j=1:b
        plot(find(y==j), 1.08*ones([1 length(find(y==j))]), '.', 'Color', colors(j,:)); % responses
    end
    title(['Response y, input u (black), and posterior expectation of reward v ', ...
           'for \alpha=', num2str(r.p_prc.al)], ...
      'FontWeight', 'bold');
    ylabel('y, u, s(\mu_2)');
    axis([0 ts(end) -0.15 1.15]);
else
    title(['Input u (black) and posterior expectation of input s(\mu_2) ', ...
           'for \alpha=', num2str(r.p_prc.al)], ...
      'FontWeight', 'bold');
    ylabel('u, s(\mu_2)');
    axis([0 ts(end) -0.1 1.1]);
end
plot(ts(2:end), 0.5, 'k');
xlabel('Trial number');
hold off;

function tapas_hgf_ar1_mab_plotTraj(r)
% Plots the estimated or generated trajectories for the HGF perceptual model
% Usage example:  est = tapas_fitModel(responses, inputs); tapas_hgf_ar1_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Optional plotting of standard deviations (true or false)
plotsd = true;

% Optional plotting of upper levels
plotul = false;

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.2*scrsz(4),0.8*scrsz(3),0.8*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name', 'HGF trajectories');

% Set up colors
colors = [1 0 0; 0.67 0 1; 0 0.67 1; 0.67 1 0];

% Number of bandits
b = r.c_prc.n_bandits;

% Number of trials
n = size(r.u,1);

% Time axis
if size(r.u,2) > 2
    t = r.u(:,end)';
else
    t = ones(1,n);
end

ts = cumsum(t);
ts = [0, ts];

if plotul == true

% Number of levels
try
    l = r.c_prc.n_levels;
catch
    l = length(r.p_prc.p)/5;
end

% Upper levels
for j = 1:l-1

    % Subplots
    subplot(l,1,j);

    if plotsd == true
        upperprior = r.p_prc.mu_0(l-j+1) +sqrt(r.p_prc.sa_0(l-j+1));
        lowerprior = r.p_prc.mu_0(l-j+1) -sqrt(r.p_prc.sa_0(l-j+1));
        upper = [upperprior; r.traj.mu(:,l-j+1,1)+sqrt(r.traj.sa(:,l-j+1,1))];
        lower = [lowerprior; r.traj.mu(:,l-j+1,1)-sqrt(r.traj.sa(:,l-j+1,1))];
    
        plot(0, upperprior, 'ob', 'LineWidth', 1);
        hold all;
        plot(0, lowerprior, 'ob', 'LineWidth', 1);
        fill([ts, fliplr(ts)], [(upper)', fliplr((lower)')], ...
             'b', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
    plot(ts, [r.p_prc.mu_0(l-j+1); r.traj.mu(:,l-j+1,1)], 'b', 'LineWidth', 2);
    hold all;
    plot(0, r.p_prc.mu_0(l-j+1), 'ob', 'LineWidth', 2); % prior
    xlim([0 ts(end)]);
    title(['Posterior expectation of x_' num2str(l-j+1)], 'FontWeight', 'bold');
    ylabel(['\mu_', num2str(l-j+1)]);
end


% Input level
subplot(l,1,l);

end % if plotul == true

if plotsd == true
    for j=1:b
        upperprior = r.p_prc.mu_0(1) +sqrt(r.p_prc.sa_0(1));
        lowerprior = r.p_prc.mu_0(1) -sqrt(r.p_prc.sa_0(1));
        upper = [upperprior; r.traj.mu(:,1,j)+sqrt(r.traj.sa(:,1,j))];
        lower = [lowerprior; r.traj.mu(:,1,j)-sqrt(r.traj.sa(:,1,j))];
    
        plot(0, upperprior, 'o', 'Color', colors(j,:), 'LineWidth', 1);
        hold all;
        plot(0, lowerprior, 'o', 'Color', colors(j,:), 'LineWidth', 1);
        fill([ts, fliplr(ts)], [(upper)', fliplr((lower)')], ...
             colors(j,:), 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
end
for j=1:b
    plot(ts, [r.p_prc.mu_0(1); r.traj.mu(:,1,j)], 'Color', colors(j,:), 'LineWidth', 2);
    hold all;
    plot(0, r.p_prc.mu_0(1), 'o', 'Color', colors(j,:), 'LineWidth', 2); % prior
end
for j=2:length(ts)
    plot(ts(j), r.u(j-1,1), '.', 'Color', colors(r.u(j-1,2),:)); % inputs
end
title(['Input u (green) and posterior expectations of x_1 ', ...
       '(red) for \phi=', num2str(r.p_prc.phi), ', m=', num2str(r.p_prc.m), ', \kappa=', ...
       num2str(r.p_prc.ka), ', \omega=', num2str(r.p_prc.om),...
      ', \alpha=', num2str(r.p_prc.al)], ...
      'FontWeight', 'bold');
ylabel('u, \mu_1');
xlim([0 ts(end)]);
xlabel({'Trial number', ' '}); % A hack to get the relative subplot sizes right
hold off;

function tapas_hgf_whichworld_plotTraj(r)
% Plots trajectories estimated by fitModel for the hgf_whichworld perceptual model
% Usage:  est = tapas_fitModel(responses, inputs); tapas_hgf_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Check whether we have a configuration structure
if ~isfield(r,'c_prc')
    error('tapas:hgf:ConfigRequired', 'Configuration required: before calling tapas_hgf_whichworld_plotTraj, tapas_hgf_whichworld_config has to be called.');
end

% Number of worlds
nw = r.c_prc.nw;

% Define colors
colors = [1 0 0; 0.67 0 1; 0 0.67 1; 0.67 1 0];

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.2*scrsz(4),0.8*scrsz(3),0.8*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name','HGF trajectories');

% Number of trials
t = size(r.u,1);

% Optional plotting of standard deviations (true or false)
plotsd2 = true;
plotsd3 = true;

% Subplots
subplot(3,1,1);

if plotsd3 == true
    upper3prior = r.p_prc.mu3_0 +sqrt(r.p_prc.sa3_0);
    lower3prior = r.p_prc.mu3_0 -sqrt(r.p_prc.sa3_0);
    upper3 = [upper3prior; r.traj.mu(:,3)+sqrt(r.traj.sa(:,3))];
    lower3 = [lower3prior; r.traj.mu(:,3)-sqrt(r.traj.sa(:,3))];
    
    plot(0, upper3prior, 'ob', 'LineWidth', 1);
    hold all;
    plot(0, lower3prior, 'ob', 'LineWidth', 1);
    fill([0:t, fliplr(0:t)], [(upper3)', fliplr((lower3)')], ...
         'b', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
end
plot(0:t, [r.p_prc.mu3_0; r.traj.mu(:,3)], 'b', 'LineWidth', 2);
hold all;
plot(0, r.p_prc.mu3_0, 'ob', 'LineWidth', 2); % prior
xlim([0 t]);
title('Posterior expectation \mu_3 of log-volatility of tendency x_3', 'FontWeight', 'bold');
xlabel('Trial number');
ylabel('\mu_3');

subplot(3,1,2);
if plotsd2 == true
    for j=1:nw
    upper2prior = r.p_prc.mu2_0(j) +sqrt(r.p_prc.sa2_0(j));
    lower2prior = r.p_prc.mu2_0(j) -sqrt(r.p_prc.sa2_0(j));
    upper2 = [upper2prior; r.traj.mu(:,2,j)+sqrt(r.traj.sa(:,2,j))];
    lower2 = [lower2prior; r.traj.mu(:,2,j)-sqrt(r.traj.sa(:,2,j))];
    
    plot(0, upper2prior, 'o', 'Color', colors(j,:), 'LineWidth', 1);
    hold all;
    plot(0, lower2prior, 'o', 'Color', colors(j,:), 'LineWidth', 1);
    fill([0:t, fliplr(0:t)], [(upper2)', fliplr((lower2)')], ...
         colors(j,:), 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
end
for j=1:nw
    plot(0:t, [r.p_prc.mu2_0(j); r.traj.mu(:,2,j)], 'Color', colors(j,:), 'LineWidth', 2);
    hold all;
    plot(0, r.p_prc.mu2_0(j), 'o', 'Color', colors(j,:), 'LineWidth', 2); % prior
end
xlim([0 t]);
title('Posterior expectations \mu_2 of tendencies x_2', 'FontWeight', 'bold');
xlabel({'Trial number', ' '}); % A hack to get the relative subplot sizes right
ylabel('\mu_2');
hold off;

subplot(3,1,3);
for j=1:nw
    plot(0:t, [tapas_sgm(r.p_prc.mu2_0(j), 1); tapas_sgm(r.traj.mu(:,2,j), 1)], 'Color', colors(j,:), 'LineWidth', 2);
    hold all;
    plot(0, tapas_sgm(r.p_prc.mu2_0(j), 1), 'o', 'Color', colors(j,:), 'LineWidth', 2); % prior
end
plot(1:t, r.u(:,1), '.', 'Color', [0 0 0]); % inputs
if ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    y = r.y(:,1);
    if ~isempty(find(strcmp(fieldnames(r),'irr')))
        y(r.irr) = NaN; % weed out irregular responses
        plot(r.irr,  1.08.*ones([1 length(r.irr)]), 'x', 'Color', [1 0.7 0], 'Markersize', 11, 'LineWidth', 2); % irregular responses
    end
    if ~any(y>1) && ~any(y<0)
        plot(1:t, y, '.', 'Color', [1 0.7 0]); % responses
    else
        for j=1:nw
            plot(find(y==j), 1.08*ones([1 length(find(y==j))]), '.', 'Color', colors(j,:)); % responses
        end
    end
    title(['Response y, input u (black), and posterior probability of worlds s(\mu_2) for \kappa=', ...
           num2str(r.p_prc.ka), ', \omega=', num2str(r.p_prc.om), ', \vartheta=', num2str(r.p_prc.th)], ...
          'FontWeight', 'bold');
    ylabel('y, u, s(\mu_2)');
    axis([0 t -0.1 1.15]);
else
    title(['Input u (black) and posterior probability of worlds s(\mu_2) for m_3=', ...
           num2str(r.p_prc.m), ', \phi_3=', num2str(r.p_prc.phi), ', \kappa=', ...
           num2str(r.p_prc.ka), ', \omega=', num2str(r.p_prc.om), ', \vartheta=', num2str(r.p_prc.th)], ...
      'FontWeight', 'bold');
    ylabel('u, s(\mu_2)');
    axis([0 t -0.1 1.1]);
end
plot(1:t, 0.5, 'k');
xlabel('Trial number');
hold off;

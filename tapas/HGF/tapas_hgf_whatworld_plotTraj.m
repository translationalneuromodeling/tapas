function tapas_hgf_whatworld_plotTraj(r)
% Plots trajectories estimated by fitModel for the hfg perceptual model
% Usage:  est = tapas_fitModel(responses, inputs); tapas_hgf_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Check whether we have a configuration structure
if ~isfield(r,'c_prc')
    error('tapas:hgf:ConfigRequired', 'Configuration required: before calling tapas_hgf_whatworld_plotTraj, hgf_whatworld_config has to be called.');
end

% Number of states
ns = r.c_prc.n_states;

% Define colors
%colors = [1 0 0; 0.67 0 1; 0 0.67 1; 0.67 1 0];

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.2*scrsz(4),0.8*scrsz(3),0.8*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name','HGF trajectories');

% Number of trials
t = size(r.u,1);

% Optional plotting of standard deviations (true or false)
plotsd1 = true;
plotsd3 = true;

% Plotting of optional transition probabilities
plotttp = false;
if size(r.u,2) > 1
    plotttp = true;
end


% Subplots
for m = 1:ns^2
    % Get subscripts for extracting elements from matrices
    % Note: columns and rows are reversed with respect ot the normal usage of ind2sub
    % because subplot plots row-wise while ind2sub works column-wise.
    [col,row] = ind2sub([ns,ns],m);
    subplot(ns+1,ns,m);

    if plotsd1 == true
        upper = tapas_sgm(r.traj.muhat(:,2,row,col)+sqrt(r.traj.sahat(:,2,row,col)),1);
        lower = tapas_sgm(r.traj.muhat(:,2,row,col)-sqrt(r.traj.sahat(:,2,row,col)),1);
        hold all;
        fill([1:t, fliplr(1:t)], [(upper)', fliplr((lower)')], ...
             'r', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
    plot(1:t, r.traj.muhat(:,1,row,col), 'Color', 'r', 'LineWidth', 2);
    hold all;
    if plotttp == true
        tm = r.u(:,2);
        ttp = NaN(t,1);
        for k = 1:t
            ttp(k) = r.c_prc.ttm(row,col,tm(k));
        end
        plot(1:t, ttp, 'Color', 'k', 'LineWidth', 1);
    end
    %title(['Posterior probability of transitions s(\mu_2) for \kappa=', ...
    %       num2str(r.p_prc.ka), ', \omega=', num2str(r.p_prc.om), ', \vartheta=', num2str(r.p_prc.th)], ...
    %      'FontWeight', 'bold');
           %ylabel('u, s(\mu_2)');
    axis([1 t -0.1 1.1]);
    hold off;
end

subplot(ns+1,ns,ns^2+1);

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
plot(0:t, [r.p_prc.mu3_0; r.traj.mu(:,3,1,1)], 'b', 'LineWidth', 2);
hold all;
plot(0, r.p_prc.mu3_0, 'ob', 'LineWidth', 2); % prior
xlim([0 t]);
title('Volatility estimate', 'FontWeight', 'bold');
%xlabel('Trial number');
ylabel('\mu_3');
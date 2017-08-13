function tapas_hgf_binary_condhalluc_plotTraj(r)
% Plots the estimated or generated trajectories for the binary HGF perceptual model
% Usage example:  est = tapas_fitModel(responses, inputs); tapas_hgf_binary_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2015 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Optional plotting of standard deviations (true or false)
plotsd = true;

% Optional plotting of responses (true or false)
ploty = true;

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.2*scrsz(4),0.8*scrsz(3),0.8*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name', 'HGF trajectories');

% Time axis
t = ones(1,size(r.u,1));

ts = cumsum(t);
ts = [0, ts];

% Number of levels
try
    l = r.c_prc.n_levels;
catch
    l = (length(r.p_prc.p)+1)/5;
end

% Upper levels
for j = 1:l-1

    % Subplots
    subplot(l,1,j);

    if plotsd == true
        upperprior = r.p_prc.mu_0(l-j+1) +sqrt(r.p_prc.sa_0(l-j+1));
        lowerprior = r.p_prc.mu_0(l-j+1) -sqrt(r.p_prc.sa_0(l-j+1));
        upper = [upperprior; r.traj.mu(:,l-j+1)+sqrt(r.traj.sa(:,l-j+1))];
        lower = [lowerprior; r.traj.mu(:,l-j+1)-sqrt(r.traj.sa(:,l-j+1))];
    
        plot(0, upperprior, 'ob', 'LineWidth', 1);
        hold all;
        plot(0, lowerprior, 'ob', 'LineWidth', 1);
        fill([ts, fliplr(ts)], [(upper)', fliplr((lower)')], ...
             'b', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
    plot(ts, [r.p_prc.mu_0(l-j+1); r.traj.mu(:,l-j+1)], 'b', 'LineWidth', 2);
    hold all;
    plot(0, r.p_prc.mu_0(l-j+1), 'ob', 'LineWidth', 2); % prior
    xlim([0 ts(end)]);
    title(['Posterior expectation of x_' num2str(l-j+1)], 'FontWeight', 'bold');
    ylabel(['\mu_', num2str(l-j+1)]);
end

% Input level
subplot(l,1,l);

plot(ts, [tapas_sgm(r.p_prc.mu_0(2), 1); tapas_sgm(r.traj.mu(:,2), 1)], 'r', 'LineWidth', 2);
hold all;
% Get true-positive rate corresponding to stimuli
tp = r.u(:,2);
mu1hat = r.traj.muhat(:,1);
% Calculate belief x using Bayes' theorem
tonebelief = tp.*mu1hat./(tp.*mu1hat + (1-mu1hat).^2);
% Belief is mu1hat in trials where there is no tone
tonebelief(find(tp==0)) = mu1hat(find(tp==0));
% Plot tone belief
plot(ts, [NaN; tonebelief], 'b', 'LineWidth', 2);
plot(0, tapas_sgm(r.p_prc.mu_0(2), 1), 'or', 'LineWidth', 2); % prior
plot(ts(2:end), r.u(:,2), '.', 'Color', [0 0.6 0]); % inputs
plot(ts(2:end), r.traj.wt(:,1), 'k') % implied learning rate 
if (ploty == true) && ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    y = r.y(:,1) -0.5; y = 1.16 *y; y = y +0.5; % stretch
    if ~isempty(find(strcmp(fieldnames(r),'irr')))
        y(r.irr) = NaN; % weed out irregular responses
        plot(ts(r.irr),  1.08.*ones([1 length(r.irr)]), 'x', 'Color', [1 0.7 0], 'Markersize', 11, 'LineWidth', 2); % irregular responses
        plot(ts(r.irr), -0.08.*ones([1 length(r.irr)]), 'x', 'Color', [1 0.7 0], 'Markersize', 11, 'LineWidth', 2); % irregular responses
    end
    plot(ts(2:end), y, '.', 'Color', [1 0.7 0]); % responses
    title(['Response y (orange), input u (green), learning rate (fine black), posterior belief (blue), and posterior expectation of input s(\mu_2) ', ...
           '(red) for \rho=', num2str(r.p_prc.rho(2:end)), ', \kappa=', ...
           num2str(r.p_prc.ka(2:end)), ', \omega=', num2str(r.p_prc.om(2:end))], ...
      'FontWeight', 'bold');
    ylabel('y, u, s(\mu_2)');
    axis([0 ts(end) -0.15 1.15]);
else
    title(['Input u (green), learning rate (fine black), posterior belief (blue), and posterior expectation of input s(\mu_2) ', ...
           '(red) for \rho=', num2str(r.p_prc.rho(2:end)), ', \kappa=', ...
           num2str(r.p_prc.ka(2:end)), ', \omega=', num2str(r.p_prc.om(2:end))], ...
      'FontWeight', 'bold');
    ylabel('u, s(\mu_2)');
    axis([0 ts(end) -0.1 1.1]);
end
plot(ts(2:end), 0.5, 'k');
xlabel('Trial number');
hold off;

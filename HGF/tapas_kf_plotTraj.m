function tapas_kf_plotTraj(r)
% Plots the estimated or generated trajectories for the Kalman filter.
% Usage example:  est = tapas_fitModel(responses, inputs); tapas_kf_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.4*scrsz(4),0.8*scrsz(3),0.6*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name', 'Kalman filter trajectory');

% Number of inputs
t = length(r.u(:,1));

% Plot
plot(0:t, [r.p_prc.mu_0; r.traj.mu], 'r', 'LineWidth', 2);
hold all;
plot(0, r.p_prc.mu_0, 'or', 'LineWidth', 2); % prior
plot(1:t, r.u(:,1), '.', 'Color', [0 0.6 0]); % inputs
if ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    plot(1:t, r.y(:,1), '.', 'Color', [1 0.7 0]); % responses
    title(['Response y (orange), input u (green), and posterior mean \mu of hidden state ', ...
           '(red) for \omega=', num2str(r.p_prc.om), ...
           ', \pi_u=', num2str(r.p_prc.pi_u)], ...
          'FontWeight', 'bold');
    ylabel('y, u, \mu');
else
    title(['Input u (green) and posterior mean \mu of hidden state ', ...
           '(red) for \omega=', num2str(r.p_prc.om), ...
           ', \pi_u=', num2str(r.p_prc.pi_u)], ...
          'FontWeight', 'bold');
    ylabel('u, \mu_1');
end
xlim([0 t]);
xlabel({'Trial number', ' '}); % A hack to get the relative subplot sizes right
hold off;

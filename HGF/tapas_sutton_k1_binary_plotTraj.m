function tapas_sutton_k1_binary_plotTraj(r)
% Plots trajectories estimated by tapas_fitModel for the tapas_rw_binary perceptual model
% Usage:  est = tapas_fitModel(responses, inputs); tapas_rw_binary_plotTraj(est);
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.7*scrsz(4),0.8*scrsz(3),0.3*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name','RW binary fit results');

% Number of trials
t = length(r.u(:,1));

% Plot
plot(0:t, [r.p_prc.vhat_1; r.traj.v], 'r', 'LineWidth', 2);
hold all;
plot(0, r.p_prc.vhat_1, 'or', 'LineWidth', 2); % prior
plot(1:t, r.u(:,1), '.', 'Color', [0 0.6 0]); % inputs
if ~isempty(find(strcmp(fieldnames(r),'y'))) && ~isempty(r.y)
    y = r.y(:,1) -0.5; y = 1.16 *y; y = y +0.5; % stretch
    if ~isempty(find(strcmp(fieldnames(r),'irr')))
        y(r.irr) = NaN; % weed out irregular responses
        plot(r.irr,  1.08.*ones([1 length(r.irr)]), 'x', 'Color', [1 0.7 0], 'Markersize', 11, 'LineWidth', 2); % irregular responses
        plot(r.irr, -0.08.*ones([1 length(r.irr)]), 'x', 'Color', [1 0.7 0], 'Markersize', 11, 'LineWidth', 2); % irregular responses
    end
    plot(1:t, y, '.', 'Color', [1 0.7 0]); % responses
    title(['Response y (orange), input u (green), and value v (red) for \mu=', ...
           num2str(r.p_prc.mu), ', Rhat=', num2str(r.p_prc.Rhat)], 'FontWeight', 'bold');
    ylabel('y, u, v');
    axis([0 t -0.15 1.15]);
else
    title(['Input u (green) and value v (red) for \mu=', ...
           num2str(r.p_prc.mu), ', Rhat=', num2str(r.p_prc.Rhat)], 'FontWeight', 'bold');
    ylabel('u, v');
    axis([0 t -0.1 1.1]);
end
plot(1:t, 0.5, 'k');
xlabel('Trial number');
hold off;

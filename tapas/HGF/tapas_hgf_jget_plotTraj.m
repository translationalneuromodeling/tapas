function tapas_hgf_jget_plotTraj(r)
% Plots the estimated trajectories for the HGF perceptual model for
% the JGET project
% Usage example:  est = tapas_fitModel(responses, inputs); tapas_hgf_plotTraj(est);
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

% Set up display
scrsz = get(0,'screenSize');
outerpos = [0.2*scrsz(3),0.2*scrsz(4),0.8*scrsz(3),0.8*scrsz(4)];
figure(...
    'OuterPosition', outerpos,...
    'Name', 'HGF trajectories');

% Time axis
if size(r.u,2) > 1 && ~isempty(find(strcmp(fieldnames(r.c_prc),'irregular_intervals'))) && r.c_prc.irregular_intervals
    t = r.u(:,end)';
else
    t = ones(1,size(r.u,1));
end

ts = cumsum(t);
ts = [0, ts];

% Do we know the generative parameters?
if size(r.u,2) > 2
    genpar = true;
    mean   = r.u(:,2);
    sd     = r.u(:,3);
else
    genpar = false;
end

% Number of levels
try
    l = r.c_prc.n_levels;
catch
    l = length(r.p_prc.p)/8;
end

% Upper levels
for j = 1:l-1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Left subplot (x)                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(l+1,2,2*j-1);

    if plotsd == true
        upperprior = r.p_prc.mux_0(l-j+1) +1.96*sqrt(r.p_prc.sax_0(l-j+1));
        lowerprior = r.p_prc.mux_0(l-j+1) -1.96*sqrt(r.p_prc.sax_0(l-j+1));
        upper = [upperprior; r.traj.mux(:,l-j+1)+1.96*sqrt(r.traj.sax(:,l-j+1))];
        lower = [lowerprior; r.traj.mux(:,l-j+1)-1.96*sqrt(r.traj.sax(:,l-j+1))];
    
        plot(0, upperprior, 'ob', 'LineWidth', 1);
        hold all;
        plot(0, lowerprior, 'ob', 'LineWidth', 1);
        fill([ts, fliplr(ts)], [(upper)', fliplr((lower)')], ...
             'b', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
    plot(ts, [r.p_prc.mux_0(l-j+1); r.traj.mux(:,l-j+1)], 'b', 'LineWidth', 1.5);
    hold all;
    plot(0, r.p_prc.mux_0(l-j+1), 'ob', 'LineWidth', 1.5); % prior
    xlim([0 ts(end)]);
    title(['Posterior expectation of x_' num2str(l-j+1)], 'FontWeight', 'bold');
    ylabel(['\mu x_', num2str(l-j+1)]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Right subplot (alpha)                  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(l+1,2,2*j);

    if plotsd == true
        upperprior = r.p_prc.mua_0(l-j+1) +1.96*sqrt(r.p_prc.saa_0(l-j+1));
        lowerprior = r.p_prc.mua_0(l-j+1) -1.96*sqrt(r.p_prc.saa_0(l-j+1));
        upper = [upperprior; r.traj.mua(:,l-j+1)+1.96*sqrt(r.traj.saa(:,l-j+1))];
        lower = [lowerprior; r.traj.mua(:,l-j+1)-1.96*sqrt(r.traj.saa(:,l-j+1))];
    
        plot(0, upperprior, 'ob', 'LineWidth', 1);
        hold all;
        plot(0, lowerprior, 'ob', 'LineWidth', 1);
        fill([ts, fliplr(ts)], [(upper)', fliplr((lower)')], ...
             'b', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    end
    plot(ts, [r.p_prc.mua_0(l-j+1); r.traj.mua(:,l-j+1)], 'b', 'LineWidth', 1.5);
    hold all;
    plot(0, r.p_prc.mua_0(l-j+1), 'ob', 'LineWidth', 1.5); % prior
    xlim([0 ts(end)]);
    title(['Posterior expectation of \alpha_' num2str(l-j+1)], 'FontWeight', 'bold');
    ylabel(['\mu \alpha_', num2str(l-j+1)]);
end


% Input level
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Left subplot (x)                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(l+1,2,2*l-1);

if plotsd == true
    upperprior = r.p_prc.mux_0(1) +1.96*sqrt(r.p_prc.sax_0(1));
    lowerprior = r.p_prc.mux_0(1) -1.96*sqrt(r.p_prc.sax_0(1));
    upper = [upperprior; r.traj.mux(:,1)+1.96*sqrt(r.traj.sax(:,1))];
    lower = [lowerprior; r.traj.mux(:,1)-1.96*sqrt(r.traj.sax(:,1))];
    
    plot(0, upperprior, 'or', 'LineWidth', 1);
    hold all;
    plot(0, lowerprior, 'or', 'LineWidth', 1);
    fill([ts, fliplr(ts)], [(upper)', fliplr((lower)')], ...
         'r', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
end
plot(ts, [r.p_prc.mux_0(1); r.traj.mux(:,1)], 'r', 'LineWidth', 1.5);
hold all;
plot(0, r.p_prc.mux_0(1), 'or', 'LineWidth', 1.5); % prior
plot(ts(2:end), r.u(:,1), '.', 'Color', [0 0.6 0]); % inputs
if genpar
    plot(ts(2:end), mean, '-', 'Color', 'k', 'LineWidth', 1); % mean of input distribution
    plot(ts(2:end), mean +1.96.*sd, '--', 'Color', 'k', 'LineWidth', 1); % 95% interval of input distribution
    plot(ts(2:end), mean -1.96.*sd, '--', 'Color', 'k', 'LineWidth', 1); % 95% interval of input distribution
end
xlim([0 ts(end)]);
title(['Input u (green) and posterior expectation of x_1 (red) for \kappa_x=', ...
       num2str(r.p_prc.kax), ', \omega_x=', num2str(r.p_prc.omx)], 'FontWeight', 'bold');
ylabel('u, \mu x_1');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Right subplot (alpha)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(l+1,2,2*l);

if plotsd == true
    upperprior = r.p_prc.mua_0(1) +1.96*sqrt(r.p_prc.saa_0(1));
    lowerprior = r.p_prc.mua_0(1) -1.96*sqrt(r.p_prc.saa_0(1));
    upper = [upperprior; r.traj.mua(:,1)+1.96*sqrt(r.traj.saa(:,1))];
    lower = [lowerprior; r.traj.mua(:,1)-1.96*sqrt(r.traj.saa(:,1))];

    transupperprior = sqrt(exp(r.p_prc.kau *upperprior +r.p_prc.omu));
    translowerprior = sqrt(exp(r.p_prc.kau *lowerprior +r.p_prc.omu));
    transupper = sqrt(exp(r.p_prc.kau *upper +r.p_prc.omu));
    translower = sqrt(exp(r.p_prc.kau *lower +r.p_prc.omu));

    plot(0, transupperprior, 'or', 'LineWidth', 1);
    hold all;
    plot(0, translowerprior, 'or', 'LineWidth', 1);
    fill([ts, fliplr(ts)], [(transupper)', fliplr((translower)')], ...
         'r', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
end
transmuaprior = sqrt(exp(r.p_prc.kau *r.p_prc.mua_0(1) +r.p_prc.omu));
plot(ts, [transmuaprior; sqrt(exp(r.p_prc.kau *r.traj.mua(:,1) +r.p_prc.omu))], 'r', 'LineWidth', 1.5);
hold all;
plot(0, transmuaprior, 'or', 'LineWidth', 1.5); % prior
if genpar
    plot(ts(2:end), sd, '--', 'Color', 'k', 'LineWidth', 1);
end
xlim([0 ts(end)]);
title(['Belief on noise (red) for \kappa_\alpha=', ...
       num2str(r.p_prc.kaa), ', \omega_\alpha=', num2str(r.p_prc.oma)], 'FontWeight', 'bold');
ylabel('\mu \alpha_1');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decision model                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(l+1,2,2*l+1);

if plotsd == true
    upper = r.traj.muxhat(:,1)+1.96*sqrt(r.p_obs.ze +r.traj.saxhat(:,1));
    lower = r.traj.muxhat(:,1)-1.96*sqrt(r.p_obs.ze +r.traj.saxhat(:,1));

    fill([ts(2:end), fliplr(ts(2:end))], [(upper)', fliplr((lower)')], ...
         'r', 'EdgeAlpha', 0, 'FaceAlpha', 0.15);
    hold all;
end
plot(ts(2:end), r.traj.muxhat(:,1), 'Color', [153/256 17/256 153/256], 'LineWidth', 1.5);
hold all;
plot(ts(2:end), r.y(:,1), '.', 'Color', [1 0.65 0], 'MarkerSize', 15); % responses
if genpar
    plot(ts(2:end), mean, '-', 'Color', 'k', 'LineWidth', 1); % mean of input distribution
    plot(ts(2:end), mean +1.96.*sd, '--', 'Color', 'k', 'LineWidth', 1); % 95% interval of input distribution
    plot(ts(2:end), mean -1.96.*sd, '--', 'Color', 'k', 'LineWidth', 1); % 95% interval of input distribution
end
xlim([1 ts(end)]);
title('Decision model: prediction of decision (purple) and decision (orange)', 'FontWeight', 'bold');
ylabel('y, \^{\mu} x_1');
xlabel({'Trial number', ' '}); % A hack to get the relative subplot sizes right
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning rate                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if genpar
subplot(l+1,2,2*l+2);
[AX, H1, H2 ] = plotyy(ts(2:end), mean, ts(2:end), r.traj.lrx(:,1));
hold all;
ylim(AX(1), [min(mean -1.96.*sd-3), max(mean +1.96.*sd+3)]);
plot(AX(1), ts(2:end), mean +1.96.*sd, '--', 'Color', 'k', 'LineWidth', 1.1); % 95% interval of input distribution
plot(AX(1), ts(2:end), mean -1.96.*sd, '--', 'Color', 'k', 'LineWidth', 1.1); % 95% interval of input distribution
set(H1, 'Color', 'k', 'LineWidth', 1.1);
set(H2, 'Color', [178/256, 34/256, 34/256], 'LineWidth', 1.5);
set(AX(1), 'YColor', 'k');
set(AX(2), 'YColor', 'k');
xlim(AX(1), [1 ts(end)]);
xlim(AX(2), [1 ts(end)]);
title('Learning rate (bordeaux) and input sampling distribution (black)', 'FontWeight', 'bold');
ylabel(AX(1), 'Input');
ylabel(AX(2), '\sigma_x');
xlabel('Trial number');
hold off;
end
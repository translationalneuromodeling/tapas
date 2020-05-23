function  fh = tapas_physio_plot_movement_outliers_fd(rp, quality_measures, ...
    censoring, censoring_threshold, verbose)
% Plots movement outliers (for censoring), based on framewise displacement
% (FD), as computed by Power et al., 2012. Also the plotting style is based
% on Power et al., 2017
%
%  fh = tapas_physio_plot_movement_outliers_fd(rp, quality_measures, ...
%           censoring, censoring_threshold)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_plot_movement_outliers_fd
%
%   See also tapas_physio_get_movement_quality_measures

% Author: Lars Kasper
% Created: 2018-02-21
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%

if nargin == 5
    % If verbose is passed as argument (from updated tapas_physio_review):
    fh = tapas_physio_get_default_fig_params(verbose);
else
    % Backwards compatibility:
    fh = tapas_physio_get_default_fig_params();
end

stringTitle = 'Model: Motion Quality Control - Framewise Displacement';
set(fh, 'Name', stringTitle);

colors = [
    1 0 0
    0 1 0
    0 0 1
    ];

%% Realignment parameter
hs(1) = subplot(3,1,1);

for iDim = 1:3
    plot(rp(:,iDim), 'Color', colors(iDim,:)); hold all;
    plot(quality_measures.rHead*rp(:,iDim+3), 'Color', colors(iDim,:), ...
        'LineStyle', ':');
end
legend('x','pitch','y','roll', 'z', 'yaw');
ylabel('mm');
set(gca,'Xticklabel',[]);
title(sprintf('Realignment Parameter (mm), rotation scaled to rHead = %d mm', ...
    quality_measures.rHead));


%% Framewise displacement and friends, subject measures
hs(2) = subplot(3,1,2);
nVols = numel(quality_measures.FD);
plot(quality_measures.absTransDisplacement, 'k'); hold all;
plot(quality_measures.absRotDisplacement, 'k--');
plot(quality_measures.FD, 'r', 'LineWidth', 4);
plot(1:nVols, ones(nVols,1)*censoring_threshold, 'r--')
legend('Absolute Transl. Displacement', 'Absolute Rot. Displacement', ...
    'FD', 'Outlier Threshold')
ylabel('mm');
set(gca,'Xticklabel',[]);
title({
    sprintf('Framewise Displacement (mm) and censoring threshold (%.1f)', ...
    censoring_threshold)
    sprintf('Mean FD: %.3f mm; RMS Movement: %.3f', ...
    quality_measures.meanFD, quality_measures.rmsMovement)
    });


%% mask of outlier regressors (stick/spike) for censoring
hs(3) = subplot(3,1,3);
imagesc(censoring.R_outlier.')


xlabel('Volume #');
title('Outlier Mask of Stick (Spike) Regressors for censored volumes');

tapas_physio_suptitle(stringTitle);

linkaxes(hs, 'x');
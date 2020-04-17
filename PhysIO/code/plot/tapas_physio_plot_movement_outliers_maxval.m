function fh = tapas_physio_plot_movement_outliers_maxval(...
    rp, quality_measures, censoring, censoring_threshold, verbose)
% Plots output figure of motion censoring based on maximum value for
% outliers of each
%
%   fh = tapas_physio_plot_movement_outliers_maxval(...
%       rp, quality_measures, censoring, censoring_threshold)
%
% IN
%   rp              [nScans,6] realignment parameter matrix
%   quality_measures. 
%                   as in tapas_physio_get_movement_quality_measures
%                   needs at least the following fields
%       rmsdTrans
%       rmsdRot
%
%   censoring.      as in tapas_physio_create_movement_regressors
%                   needs at least the following fields
%       iOutlierTrans   [1, nOutliersTrans] volume indices of high translation
%                   differences (outliers) in rp
%       iOutlierRot     [1, nOutliersRot] volume indices of high rotational
%                   differences (outliers) in rp
%
%   censoring_threshold
%                   [1, 1...6] censoring thresholds for determining high
%                   values of dR as outliers
%                   1 value for shared trans/rot (mm/deg) maximum
%                   2 values for separate trans/rot maxima
%                   6 values for one threshold for each
%                   translation/rotation axis (x,y,z,pitch,roll,yaw)
% OUT
%   fh              figure handle of output figure
% EXAMPLE
%   tapas_physio_plot_movement_outliers_maxval
%
%   See also

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

iOutlierTrans   = censoring.iOutlierTrans;
iOutlierRot     = censoring.iOutlierRot;
rmsdTrans       = quality_measures.rmsdTrans;
rmsdRot         = quality_measures.rmsdRot;
t               = 1:size(rp,1);

switch numel(censoring_threshold)
    case {1,2}
        outlier_translation_mm = censoring_threshold(1);
        outlier_rotation_deg = censoring_threshold(end);
    case 6
        outlier_translation_mm = censoring_threshold(1:3);
        outlier_rotation_deg = censoring_threshold(4:end);
end

nOutlierTrans = numel(iOutlierTrans);
nOutlierRot = numel(iOutlierRot);

stringTitle = 'Model: Motion Quality Control - MAXVAL thresholds per direction';

if nargin == 5
    % If verbose is passed as argument (from updated tapas_physio_review):
    fh = tapas_physio_get_default_fig_params(verbose);
else
    % Backwards compatibility:
    fh = tapas_physio_get_default_fig_params();
end

set(fh, 'Name', stringTitle);

subplot(2,1,1);
plot(rp(:,1:3),'-'); hold on
plot(rmsdTrans, 'k-');
for d = 1:numel(outlier_translation_mm)
    plot(t, ones(size(t))*outlier_translation_mm(d), 'r--');
    if nOutlierTrans
        hl = stem(iOutlierTrans, outlier_translation_mm(d)*ones(1, nOutlierTrans));
        set(hl, 'Color', [1 0 0], 'LineWidth', 3);
    end
end

legend('shift x (mm)', 'shift y (mm)', 'shift z (mm)', 'RMS diff translation (mm)', ...
    sprintf('outlier threshold (%.1f mm)', outlier_translation_mm(1)), ...
    'excess translation volumes');
xlabel('scans');
ylabel('translation (mm)');

subplot(2,1,2);
plot(rp(:,4:6)*180/pi,'-'); hold on
plot(rmsdRot*180/pi, 'k-');
for d = 1:numel(outlier_rotation_deg)
    plot(t, ones(size(t))*outlier_rotation_deg(d), 'r--');
    if nOutlierRot
        hl = stem(iOutlierRot, outlier_rotation_deg(d)*ones(1, nOutlierRot));
        set(hl, 'Color', [1 0 0], 'LineWidth', 3);
    end
end

legend('pitch x (deg)', 'roll y (deg)', 'yaw z (deg)', 'RMS diff rotation (deg)', ...
    sprintf('outlier threshold (%.1f deg)', outlier_rotation_deg(1)), ...
    'excess rotation volumes');

xlabel('scans');
ylabel('rotation (deg)');
tapas_physio_suptitle(stringTitle);
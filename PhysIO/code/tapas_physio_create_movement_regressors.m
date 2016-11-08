function [R, verbose] = tapas_physio_create_movement_regressors(movement, verbose)
% Reads realignment parameters, creates derivative/squared & outlier regressor
%
% [R, verbose] = tapas_physio_create_movement_regressors(movement, verbose)
%
% The 6 realignment parameters can be augmented by their derivatives (in
% total 12 parameters) + the squares of parameters and derivatives (in
% total 24 parameters), leading to a Volterra expansion as in
%
%               Friston KJ, Williams S, Howard R, Frackowiak
%               RS, Turner R. Movement-related effects in fMRI
%               time-series. Magn Reson Med. 1996;35:346?355.)
%
% IN
%   movement    physio.model.movement
%   verbose     physio.verbose
% OUT
%   R           [nScans, (6|12|24)+nOutliers] regressor matrix from movement
%
% EXAMPLE
%   tapas_physio_create_movement_regressors
%
%   See also tapas_physio_new tapas_physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2015-07-10
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id$


[fp, fn, fs] = fileparts(movement.file_realignment_parameters);

if ~exist(movement.file_realignment_parameters, 'file')
    verbose = tapas_physio_log('No input multiple regressors found', verbose, 1);
    R = [];
else
    rp = load(movement.file_realignment_parameters);
    if strcmp('.txt', fs) % text-file
        R = rp;
    else % mat file
        R = rp.R;
    end
    
    % Include derivatives (needed later for outliers as well
    
    dR= diff(R);
    dR = [zeros(1,size(R,2)); dR];
    
    if movement.order > 6
        R = [R, dR];
    end
    
    
    % Include squared regressors/derivatives
    
    if movement.order > 12
        R = [R , R.^2];
    end
    
    
    % Include outlier movements exceeding thresholds as stick regressors
    % Euclidean distance used!
    
    sumTrans = sqrt(sum(dR(:,1:3).^2,2));
    sumRot = sqrt(sum(dR(:,4:6).^2,2));
    iOutlierTrans = find(sumTrans > movement.outlier_translation_mm);
    iOutlierRot = find( sumRot > movement.outlier_rotation_deg/180*pi);
    
    nOutlierTrans = numel(iOutlierTrans);
    nOutlierRot = numel(iOutlierRot);
    if verbose.level > 2
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params;
        subplot(2,1,1);
        plot(R(:,1:3),'-'); hold on
        plot(sumTrans, '-');
        if nOutlierTrans
            stem( iOutlierTrans,...
                movement.outlier_translation_mm*ones(1,nOutlierTrans))
        end
        legend('shift x (mm)', 'shift y (mm)', 'shift z (mm)', 'RMS diff translation (mm)', ...
            'excess translation volumes');
        xlabel('scans');
        ylabel('translation (mm)');
        
        subplot(2,1,2);
        plot(R(:,4:6)*180/pi,'-'); hold on
        plot(sumRot*180/pi, '-');
        if nOutlierRot
            stem(iOutlierRot,...
                movement.outlier_rotation_deg*ones(1,nOutlierRot))
        end
        legend('pitch x (deg)', 'roll y (deg)', 'yaw z (deg)', 'RMS diff rotation (deg)', ...
            'excess rotation volumes');
        
        xlabel('scans');
        ylabel('rotation (deg)');
    end
    
    iOutlierArray = unique([iOutlierTrans; iOutlierRot]);
    nOutliers = numel(iOutlierArray);
    
    nScans = size(R,1);
    R_outlier = zeros(nScans, nOutliers);
    for iOutlier = 1:nOutliers
        R_outlier(iOutlierArray(iOutlier), iOutlier) = 1;
    end
    
    R = [R, R_outlier];
    
end

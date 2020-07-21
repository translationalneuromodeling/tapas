function [R, movement, verbose] = tapas_physio_create_movement_regressors(...
    movement, verbose)
% Reads realignment parameters, creates derivative/squared & outlier regressor
%
% [R, movement, verbose] = ...
%       tapas_physio_create_movement_regressors(movement, verbose)
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
%   R                   [nScans, (6|12|24)+nOutliers] regressor matrix 
%                       from movement modeling
%   movement            structure (as defined in tapas_physio_new) with the
%                       following fields
%   rp                  [nScans, 6] realignment parameters
%   dRp                 [nScans, 6] temporal difference of realignment
%                       parameters, prepending one line of zeros (for 1st
%                       scan)
%   quality_measures.   structure holding the suggested quality control
%                       measures for subject motion of Power et al., 2014,
%                       Fig. 2; all rotational parameters are transformed
%                       into translations by multiplying with rHead (arc
%                       length)
%       FD              framewise displacement (FD)
%       absTransDisplacement
%                       sum of absolute values of translational (x,y,z)
%                       realignment estimates, reflecting absolute
%                       displacement of the head
%       absRotDisplacement
%                       sum of absolute values of rotational (rotation 
%                       around x,y,z-axis = pitch/roll/yaw)
%                       realignment estimates, reflecting absolute
%                       displacement of the head on its surface (arc
%                       length)
%       rmsdTrans       root mean squared framewise translational displacement (over x,y,z)
%                       equals to euclidean distance of center of
%                       mass of head between consecutive volumes
%       rmsdRot         root mean squared framewise rotational displacement 
%                       (rotation around x,y,z axis)
%                       equals to euclidean distance of between angles
%                       between consecutive volumes
%       meanFD          mean (over volumes) of framewise displacement 
%                       summary measure for subject (or session)
%       rmsMovement     root mean square (over scans) of detrended
%                       realignment estimates
%                       summary measure for subject (or session)
%
%   censoring.          structure with censoring information (detected outliers,
%                       computed quality values for censoring method maxval/fd:  
%       nOutliers       number of detected outliers
%       R_outlier       stick/spike regressors indicating location of
%                       detected outliers
%       iOutlierTrans   volume indices of translation-related outliers
%       iOutlierRot     volume indices of rotation-related outliers
%       iOutlierArray   volume indices of all detected outliers
%
%
% EXAMPLE
%   tapas_physio_create_movement_regressors
%
%   See also tapas_physio_new tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2015-07-10
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

rHead = 50; % head radius in mm for FD computation (Power et al., 2012)

[fp, fn, ext] = fileparts(movement.file_realignment_parameters);

if ~exist(movement.file_realignment_parameters, 'file')
    verbose = tapas_physio_log('No input multiple regressors found', verbose, 1);
    R = [];
else
    tmp = load(movement.file_realignment_parameters);
    if strcmp('.txt', ext) % text-file
        rp = tmp;
    else % mat file
        rp = tmp.R;
    end
    
    
    %% Motion Quality control
    
     [quality_measures, dRp] = ...
         tapas_physio_get_movement_quality_measures(rp, rHead);
     
    
    %% Motion 6/12/24 model
    R = rp;
    
    % Include derivatives
    if movement.order > 6
        R = [rp, dRp];
    end
    
    % Include squared regressors/derivatives
    if movement.order > 12
        R = [R, R.^2];
    end
    
    % Sanity check if order misspecified
    movement.order = size(R, 2);
    
    %% Motion Censoring
    % Include outlier movements exceeding thresholds as stick regressors
    % Euclidean distance used!
    iOutlierArray = [];
    iOutlierTrans = [];
    iOutlierRot = [];
    
    
 
    switch lower(movement.censoring_method)
        case 'none' % done
        case 'fd' % framewise displacement
            iOutlierArray = find(quality_measures.FD > ...
                movement.censoring_threshold);
        case 'maxval'   % tresholds for max abrupt translation/rotation,
                        % even per axis
            switch numel(movement.censoring_threshold)
                case {1,2}
                    outlier_translation_mm = movement.censoring_threshold(1);
                    outlier_rotation_deg = movement.censoring_threshold(end);
                    iOutlierTrans = find(quality_measures.rmsdTrans > outlier_translation_mm);
                    iOutlierRot = find(quality_measures.rmsdRot > outlier_rotation_deg/180*pi);
                case 6
                    ct = movement.censoring_threshold;
                    ct(4:6) = ct/180*pi;
                    iOutlierTrans = dRp(:,1) > ct(1) || dRp(:,2) > ct(2) || dRp(:,3) > ct(3);
                    iOutlierRot = dRp(:,4) > ct(4) || dRp(:,5) > ct(5) || dRp(:,6) > ct(6);
                otherwise
                    error('censoring threshold has to be 1,2 or 6 element vector. See tapas_physio_new');
            end
             
            iOutlierArray = unique([iOutlierTrans; iOutlierRot]);
        case 'dvars' %DVARS, as in Power et al, 2012
            % TODO
    end
    
    
    %% Construct censoring regressor matrix
    nOutliers = numel(iOutlierArray);
    
    nScans = size(rp,1);
    R_outlier = zeros(nScans, nOutliers);
    for iOutlier = 1:nOutliers
        R_outlier(iOutlierArray(iOutlier), iOutlier) = 1;
    end
    
   
    censoring = struct('nOutliers', nOutliers, 'R_outlier', R_outlier, ...
        'iOutlierTrans', iOutlierTrans, 'iOutlierRot', iOutlierRot, ...
        'iOutlierArray', iOutlierArray);
    
    if verbose.level >= 2
        switch lower(movement.censoring_method)
            case 'fd'
                verbose.fig_handles(end+1) = tapas_physio_plot_movement_outliers_fd(rp, ...
                    quality_measures, censoring, movement.censoring_threshold);
            case 'maxval'
                verbose.fig_handles(end+1) = tapas_physio_plot_movement_outliers_maxval(rp, ...
                    quality_measures, censoring, movement.censoring_threshold);
        end
    end
    
    
    %% Gather return values
    movement.censoring = censoring;
    movement.rp = rp;
    movement.dRp = dRp;
    movement.quality_measures = quality_measures;
    
    R = [R, R_outlier];
    
end

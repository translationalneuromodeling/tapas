function    [quality_measures, dRp] = ...
    tapas_physio_get_movement_quality_measures(R, rHead)
% Computes movement quality measures (FD, rmsMovement etc), as described in 
% Power et al., 2014
%
%   [quality_measures, dR] = tapas_physio_get_movement_quality_measures(R, rHead);
%
% IN
%   R       [nScans,6]  realignment parameters estimates in mm and rad
%                       as output by SPM (x,y,z,pitch,roll,yaw)
%   rHead               head radius in mm (default: 50 mm)
%
% OUT
%   quality_measures    structure holding the suggested quality control
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
%   dRp      [nScans,6] temporal difference of R (first value set to 0)
%
% EXAMPLE
%   tapas_physio_get_movement_quality_measures
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

dRp = diff(R);
dRp = [zeros(1,size(R,2)); dRp];

% summary motion values
rmsdTrans = sqrt(sum(dRp(:,1:3).^2,2));
rmsdRot = sqrt(sum(dRp(:,4:6).^2,2));

% from Power et al., 2014, Fig. 2
FD                      = sum(abs(dRp(:,1:3)),2) + rHead*sum(abs(dRp(:,4:6)),2);
absTransDisplacement    = sum(abs(R(:,1:3)),2);
absRotDisplacement      = rHead*sum(abs(R(:,4:6)),2);
meanFD                  = mean(FD);
detrendedR              = detrend(R); % TODO: replace by simple X\y
detrendedR(:,4:6)       = rHead*detrendedR(:,4:6);

% not sure whether Power meant indeed averaging over 6 estimates per time point as well...
%rmsMovement             = sqrt(mean(mean(detrendedR.^2,2),1));
% this fits the values in his paper better:
rmsMovement             = sqrt(mean(sum(abs(detrendedR),2).^2,1));

quality_measures = struct('FD', FD, ...
    'absTransDisplacement', absTransDisplacement, ...
    'absRotDisplacement', absRotDisplacement, ...
    'rmsdTrans', rmsdTrans, 'rmsdRot', rmsdRot, ...
    'meanFD', meanFD, 'rmsMovement', rmsMovement, 'rHead', rHead);


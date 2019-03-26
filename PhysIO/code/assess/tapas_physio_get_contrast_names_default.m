function namesPhysContrasts = tapas_physio_get_contrast_names_default()
% Returns cell of defaults names for phys contrasts used e.g. in reporting
%
%   namesPhysContrasts = tapas_physio_get_contrast_names_default()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_get_contrast_names_default
%
%   See also tapas_physio_report_contrasts tapas_physio_compute_tsnr_gains

% Author: Lars Kasper
% Created: 2016-10-03
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


namesPhysContrasts = {
    'All Phys'
    'Cardiac'
    'Respiratory'
    'Card X Resp Interation'
    'HeartRateVariability'
    'RespiratoryVolumePerTime'
    'Noise Rois'
    'Movement'
    'All Phys + Move'
    };
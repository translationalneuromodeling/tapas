function physio = tapas_physio_fill_empty_parameters(physio)
%fills empty values with default parameters, typically derived from other
%parameters, e.g. time_slice_to_slice from TR and Nslices
%
% NOTE: for real default values that do not depend on other input
% parameters, set them in tapas_physio_new directly!
%
%  physio = tapas_physio_fill_empty_parameters(physio)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_fill_empty_parameters
%
%   See also tapas_physio_new tapas_physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2014-04-27
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 464 2014-04-27 11:58:09Z kasperla $

if isempty(physio.sqpar.NslicesPerBeat)
    physio.sqpar.NslicesPerBeat = physio.sqpar.Nslices;
end

if isempty(physio.sqpar.time_slice_to_slice)
    physio.sqpar.time_slice_to_slice = physio.sqpar.TR/physio.sqpar.Nslices;
end

if isempty(physio.log_files.sampling_interval)
    switch lower(physio.log_files.vendor)
        case 'philips'
            physio.log_files.sampling_interval = 2e-3;
        case 'siemens'
            physio.log_files.sampling_interval = 1/400;
        case 'ge'
            physio.log_files.sampling_interval = 25e-3;
        otherwise % e.g. custom
            physio.log_files.sampling_interval = 25e-3;
    end
end
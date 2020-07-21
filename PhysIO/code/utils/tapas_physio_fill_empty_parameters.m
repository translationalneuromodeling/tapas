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

% Author: Lars Kasper
% Created: 2014-04-27
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



if isempty(physio.scan_timing.sqpar.NslicesPerBeat)
    physio.scan_timing.sqpar.NslicesPerBeat = physio.scan_timing.sqpar.Nslices;
end

if isempty(physio.scan_timing.sqpar.Ndummies)
    physio.scan_timing.sqpar.Ndummies = 0;
end

if isempty(physio.scan_timing.sqpar.time_slice_to_slice)
    physio.scan_timing.sqpar.time_slice_to_slice = physio.scan_timing.sqpar.TR/physio.scan_timing.sqpar.Nslices;
end

if isempty(physio.log_files.scan_timing)
    physio.log_files.scan_timing = {''};
end

if strcmpi(physio.preproc.cardiac.initial_cpulse_select.method, 'auto_matched') && ...
    isempty(physio.preproc.cardiac.initial_cpulse_select.min)
    physio.preproc.cardiac.initial_cpulse_select.min = 0.4;
end

if isempty(physio.log_files.sampling_interval)
    switch lower(physio.log_files.vendor)
        case {'bids', 'biopac_mat', 'brainproducts', 'siemens'} % will be read from file later
            physio.log_files.sampling_interval = [];
        case 'biopac_txt'
            physio.log_files.sampling_interval = 1/1000;
        case 'ge'
            physio.log_files.sampling_interval = 25e-3;
        case 'philips'
            isWifi      = regexpi(physio.preproc.cardiac.modality, '_wifi');
            % different sampling interval for Wifi devices
            if isWifi
                physio.log_files.sampling_interval = 1/496;
            else
                physio.log_files.sampling_interval = 1/500;
            end
        case {'siemens_tics', 'siemens_hcp'}
             physio.log_files.sampling_interval = 2.5e-3;
        otherwise % e.g. custom
            error('Please specify sampling interval for custom text data');
    end
end
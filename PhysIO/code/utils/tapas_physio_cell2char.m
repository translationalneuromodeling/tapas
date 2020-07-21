function physio = tapas_physio_cell2char(physio)
%convert all cellstrings (e.g. created via matlabbatch) back into regular
%strings for all relevant parameters of physio-structure
%
%    physio = tapas_physio_cell2char(physio)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_cell2char
%
%   See also

% Author: Lars Kasper
% Created: 2014-04-28
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


physio.save_dir = char(physio.save_dir);
physio.log_files.cardiac = char(physio.log_files.cardiac);
physio.log_files.respiration = char(physio.log_files.respiration);
physio.log_files.scan_timing = char(physio.log_files.scan_timing);
physio.model.movement.file_realignment_parameters = char(...
    physio.model.movement.file_realignment_parameters);
physio.model.other.input_multiple_regressors = char(...
    physio.model.other.input_multiple_regressors);
physio.model.output_multiple_regressors = char(...
    physio.model.output_multiple_regressors);
physio.model.output_physio = char(...
    physio.model.output_physio);
physio.verbose.fig_output_file = char(physio.verbose.fig_output_file);
function physio = tapas_physio_prepend_absolute_paths(physio)
%prepend absolute paths for file names, in particular save_dir
%
%   physio = tapas_physio_prepend_absolute_paths(physio)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_prepend_absolute_paths
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-05-03
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 464 2014-04-27 11:58:09Z kasperla $

save_dir = physio.save_dir;

if ~isequal(save_dir, fileparts(physio.verbose.fig_output_file))
    physio.verbose.fig_output_file = fullfile(save_dir, ...
        physio.verbose.fig_output_file);
end

if ~isequal(save_dir, fileparts(physio.model.output_multiple_regressors))
    physio.model.output_multiple_regressors = fullfile(save_dir, ...
        physio.model.output_multiple_regressors);
end

if ~isequal(save_dir, fileparts(physio.thresh.cardiac.initial_cpulse_select.file))
    physio.thresh.cardiac.initial_cpulse_select.file = fullfile(save_dir, ...
        physio.thresh.cardiac.initial_cpulse_select.file);
end

if ~isequal(save_dir, fileparts(physio.thresh.cardiac.posthoc_cpulse_select.file))
    physio.thresh.cardiac.posthoc_cpulse_select.file = fullfile(save_dir, ...
        physio.thresh.cardiac.posthoc_cpulse_select.file);
end
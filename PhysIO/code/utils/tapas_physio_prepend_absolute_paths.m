function physio = tapas_physio_prepend_absolute_paths(physio)
% prepend absolute paths for file names, in particular save_dir; creates
% save_dir, if necessary
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

% Author: Lars Kasper
% Created: 2014-05-03
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


[parentPath, currentFolder] = fileparts(physio.save_dir);

%% Make all folders in physio struct absolute
% only relative folder for save_dir specified, make absolute
if (isempty(parentPath) || ( iscell(parentPath) && isempty([parentPath{:}]))) && ~isempty(currentFolder)
    physio.save_dir = fullfile(pwd, physio.save_dir);
end

[parentPath, currentFolder] = fileparts(physio.write_bids.bids_dir);

% only relative folder specified, make absolute
if (isempty(parentPath) || ( iscell(parentPath) && isempty([parentPath{:}]))) && ~isempty(currentFolder)
    physio.write_bids.bids_dir = fullfile(pwd, physio.write_bids.bids_dir);
end

%% update all affected relative file names with save_dir

requiredDirs = {physio.save_dir, physio.write_bids.bids_dir};

for iRequiredDirs = 1:numel(requiredDirs)

    required_dir = requiredDirs{iRequiredDirs};

    if ~exist(required_dir, 'dir') && ~isempty(required_dir)
        [status,msg,msgID] = mkdir(required_dir);
        if ~status % unsuccessful creation AND directory did not exist before
            physio.verbose = tapas_physio_log(msg, physio.verbose, 2);
        end
    end
end

save_dir = physio.save_dir;

if ~isequal(save_dir, fileparts(physio.verbose.fig_output_file))
    physio.verbose.fig_output_file = fullfile(save_dir, ...
        physio.verbose.fig_output_file);
end

if ~isequal(save_dir, fileparts(physio.model.output_multiple_regressors))
    physio.model.output_multiple_regressors = fullfile(save_dir, ...
        physio.model.output_multiple_regressors);
end

if ~isequal(save_dir, fileparts(physio.model.output_physio))
    physio.model.output_physio = fullfile(save_dir, ...
        physio.model.output_physio);
end

if ~isequal(save_dir, fileparts(physio.preproc.cardiac.initial_cpulse_select.file))
    physio.preproc.cardiac.initial_cpulse_select.file = fullfile(save_dir, ...
        physio.preproc.cardiac.initial_cpulse_select.file);
end

if ~isequal(save_dir, fileparts(physio.preproc.cardiac.posthoc_cpulse_select.file))
    physio.preproc.cardiac.posthoc_cpulse_select.file = fullfile(save_dir, ...
        physio.preproc.cardiac.posthoc_cpulse_select.file);
end

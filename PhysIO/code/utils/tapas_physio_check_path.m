function [isPhysioOnPath, pathPhysIO] = tapas_physio_check_path()
% Checks whether physio is on path and returns suggested path for adding
%
%  [isPhysioOnPath, pathPhysIO] = tapas_physio_check_path()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_check_path
%
%   See also

% Author: Lars Kasper
% Created: 2018-02-17
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%

% 2x fileparts, since it is the parent folder of where check_path resides
pathPhysIO = fileparts(fileparts(mfilename('fullpath')));

% cell of all subfolders of PhysIO/code, remove empty paths, e.g. due to
% trailing pathsep
pathPhysIOAll = regexp(genpath(pathPhysIO), pathsep, 'split');
pathPhysIOAll(cellfun(@isempty, pathPhysIOAll)) = [];
% cell of all folders in Matlab path
pathCell = regexp(path, pathsep, 'split');

% check for all entries of physio-code whether in path
isPhysioOnPath = true;

n = 0;
nPaths = numel(pathPhysIOAll);

while isPhysioOnPath && n < nPaths
    n = n + 1;
    if ispc || ismac % Windows/Mac is not case-sensitive
        isPhysioOnPath = any(strcmpi(pathPhysIOAll{n}, pathCell));
    else
        isPhysioOnPath = any(strcmp(pathPhysIOAll{n}, pathCell));
    end
end
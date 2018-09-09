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
%
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
pathPhysIO = fileparts(mfilename('fullpath'));

pathCell = regexp(path, pathsep, 'split');
if ispc || ismac % Windows/Mac is not case-sensitive
    isPhysioOnPath = any(strcmpi(pathPhysIO, pathCell));
else
    isPhysioOnPath = any(strcmp(pathPhysIO, pathCell));
end
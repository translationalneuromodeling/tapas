function fpFile = tapas_physio_filename2path(filename)
% Converts string (also within a cell) of filename in an absolute path,
% leaves it unaltered, if it is one already
%
%   output = tapas_physio_filename2path(input)
%
% IN
%   filename    (cell) string of filename, with or without absolute path
% OUT
%   fpFile      string of filename with prepended absolute path 
%               (pwd is prepended)
% EXAMPLE
%   tapas_physio_filename2path
%
%   See also

% Author: Lars Kasper
% Created: 2016-10-03
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% strip cell
if iscell(filename)
    fpFile = filename{1};
else
    fpFile = filename;
end

% check whether absolute path, i.e. starting with / (Linux/Max) 
% or <Disk Letter>: )(Windows)
fp = fileparts(fpFile);
if isempty(fp) || (~ispc && fp(1) ~= '/') || (ispc && fp(2) ~= ':')
    fpFile = fullfile(pwd, fpFile);
end

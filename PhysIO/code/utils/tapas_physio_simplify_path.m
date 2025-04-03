function pathName = tapas_physio_simplify_path(pathName)
% Simplifies path name string by replacing all relative paths in it, i.e. ..\, .\
% NEW version, used Matlab's dir functionality
%  pathName = tapas_uniqc_simplify_path(pathName)
%
% IN
%   pathName    string with relative path symbols, e.g.
%               'C:\bla\recon\test\..\code\classes\..\..\test\.\'
% OUT
%   pathName    string, simplified path name, e.g.
%               'C:\bla\recon\test\'
%
% EXAMPLE
%   tapas_physio_simplify_path
%
%   See also
 
% Author:   Lars Kasper
% Created:  2023-11-13
% Copyright (C) 2023 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
outputDirs = dir(pathName);
pathName = outputDirs(1).folder;
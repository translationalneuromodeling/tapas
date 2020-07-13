function [isSpmOnPath, pathSpm] = tapas_physio_check_spm();
% Checks whether SPM exists in path, using also its check_installation for
% correct subfolder addpath
%
%   [isSpmOnPath, pathSpm] = tapas_physio_check_spm();
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_check_spm
%
%   See also spm_check_installation tapas_physio_check_spm_batch_editor_integration

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

isSpmOnPath = (exist('spm') == 2);
if isSpmOnPath
    pathSpm = spm('Dir'); 
    spm_check_installation();
else
    warning(sprintf(...
        [' SPM is not on your Matlabpath. Please add it without its subfolders,', ...
         '\n e.g., via addpath, if you want to use PhysIO with the SPM Batch Editor GUI']));
    pathSpm = '';
end
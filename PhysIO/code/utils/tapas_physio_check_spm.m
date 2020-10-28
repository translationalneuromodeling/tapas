function [isSpmOnPath, pathSpm] = tapas_physio_check_spm(doCorrectPath)
% Checks whether SPM exists in path, using also its check_installation for
% correct subfolder addpath
%
%   [isSpmOnPath, pathSpm] = tapas_physio_check_spm();
%
% IN
%   doCorrectPath   if true, erroneous subfolders from SPM will be removed
%                   from path (apart from PhysIO); default: false
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

if nargin < 1
    doCorrectPath = false;
end

isSpmOnPath = (exist('spm') == 2);
if isSpmOnPath
    pathSpm = spm('Dir'); 
    spm_check_installation();
    % as in spm_check_installation, check whether a field trip function is 
    % on the path that should never by, and we rectify the installation by removing
    % all SPM subfolders and adding only SPM itself plus toolbox/PhysIO
    if doCorrectPath && exist('ft_check_path') % this should not exist
       warning(['Removing subfolders of SPM from Matlab path to avoid' ...
            ' errors in PhysIO related to overloading functions from fieldtrip']); 
       pathPhysio = fileparts(fileparts(mfilename('fullpath'))); % utils/../ folder, should evaluate to PhysIO/code
       warning('off', 'MATLAB:rmpath:DirNotFound')
       rmpath(genpath(pathSpm));
       warning('on', 'MATLAB:rmpath:DirNotFound')
       addpath(pathSpm);
       % re-add PhysIO to path recursively, if it resides within SPM folder,
       % since it was removed by spm_rmpath
       wasPhysioFolderWithinSpm = any(strfind(pathPhysio, pathSpm));
       if wasPhysioFolderWithinSpm
           addpath(genpath(pathPhysio));
       end
    end
else
    warning(sprintf(...
        [' SPM is not on your Matlabpath. Please add it WITHOUT its subfolders,', ...
         '\n e.g., via addpath, if you want to use PhysIO with the SPM Batch Editor GUI']));
    pathSpm = '';
end
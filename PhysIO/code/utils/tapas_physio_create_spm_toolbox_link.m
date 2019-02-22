function cmdString = tapas_physio_create_spm_toolbox_link(pathPhysIO)
% Creates a symbolik link of PhysIO/code folder to subfolder SPM/toolbox/PhysIO
% to make toolbox visible to SPM Batch editor
%
%   tapas_physio_create_spm_toolbox_link()
%
% IN
%
% OUT
%   cmdString   string of executed command
%
% EXAMPLE
%   tapas_physio_create_spm_toolbox_link
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
cmdString = '';

if nargin < 1
    % 2x fileparts, because parent folder
    pathPhysIO = fileparts(fileparts(mfilename('fullpath')));
end

pathSpm = fileparts(which('spm'));

if isempty(pathSpm)
    warning('SPM folder not found. Could not create symbolic link to PhysIO Toolbox');
else
    pathLinkPhysIOSPM = fullfile(pathSpm, 'toolbox', 'PhysIO');
    if ~exist(pathLinkPhysIOSPM, 'dir')
        % Create Link or hard-copy folder, OS dependent
        if ispc
            % unfortunately, system link does not work for SPM, has to be
            % hard copy
            fprintf('Copying %s to %s, because symlink not sufficient on Windows...\n', pathPhysIO, pathLinkPhysIOSPM);
            cmdString = sprintf('copyfile(''%s'', ''%s'');',pathPhysIO, pathLinkPhysIOSPM);
            eval(cmdString);
            % linking indeed the other way around than in Linux/Mac
            % cmdString = sprintf('mklink /D %s %s', pathLinkPhysIOSPM, pathPhysIO);
        else %unix/Mac
            cmdString = sprintf('ln -s %s %s', pathPhysIO, pathLinkPhysIOSPM);
            system(cmdString);
        end
    else
        warning('Destination spm/toolbox folder %s already found. Will not overwrite or re-link', pathLinkPhysIOSPM);
    end
end
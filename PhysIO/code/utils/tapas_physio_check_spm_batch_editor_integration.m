function [isPhysioVisibleForSpmBatchEditor, pathSpm, pathPhysIO] = ...
    tapas_physio_check_spm_batch_editor_integration(isVerbose)
% Checks whether PhysIO-configuration file for matlabbatch is in subfolder
% of SPM/toolbox and returns a warning, if not
%
%   [isPhysioVisibleForSpmBatchEditor, pathSpm, pathPhysIO] = ...
%       tapas_physio_check_spm_batch_editor_integration()
%
% IN
%   isVerbose   if true, warning is issued
% OUT
%
% EXAMPLE
%   tapas_physio_check_spm_batch_editor_integration
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

if nargin < 1
    isVerbose = true;
end

[isSpmOnPath, pathSpm] = tapas_physio_check_spm();

isPhysioVisibleForSpmBatchEditor = false; % checked below, need SPM visible for that

% check for config matlabbatch file
if isSpmOnPath
    
    % check all possible SPM toolbox locations, as listed in its defaults:
    tbx = spm_get_defaults('tbx'); % SPM toolbox parameter struct
    
    iDir = 1;
    while (iDir <= numel(tbx.dir)) && ~isPhysioVisibleForSpmBatchEditor 
        filePhysioCfgMatlabbatch = ...
            dir(fullfile(tbx.dir{iDir}, '**/tapas_physio_cfg_matlabbatch.m'));
        
        isPhysioVisibleForSpmBatchEditor = ~isempty(filePhysioCfgMatlabbatch);
        iDir = iDir + 1;
    end
    
    % also important to set default modality of spm to fMRI and
    % initialize batch editor, if not done before
    spm('defaults', 'fmri');
    if ~exist('cfg_util')
        spm_jobman('initcfg');
    end
end

[isPhysioOnPath, pathPhysIO] = tapas_physio_check_path();

if ~isPhysioVisibleForSpmBatchEditor && isVerbose
    warning(['\n The PhysIO Toolbox code folder has not been copied (or linked)' ...
        ' to a subfolder of the SPM/toolbox folder. \n The Batch Editor will' ...
        ' not show PhysIO. \n To make PhysIO visible there, link its path' ...
        ' by typing the following in the Matlab command window (Linux/Mac) \n ' ...
        ' and restart SPM afterwards:\n' ...
        '       ln -s %s %s/toolbox/PhysIO'], pathPhysIO, pathSpm);
end
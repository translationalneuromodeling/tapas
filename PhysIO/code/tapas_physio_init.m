function isPhysioCorrectlyInitialized = tapas_physio_init()
% Initializes TAPAS by checking that all folders are at the right
%
%    isPhysioCorrectlyInitialized = tapas_physio_init()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_init()
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

% add path for utils, if physio not in path
if ~exist('tapas_physio_logo')
    pathPhysio = fileparts(mfilename('fullpath'));
    addpath(fullfile(pathPhysio, 'utils')); % needed for further path checks
end

tapas_physio_logo(); % print logo

disp('Checking Matlab PhysIO paths, SPM paths and Batch Editor integration now...');

%% Check and add physio paths
fprintf('Checking Matlab PhysIO paths now...');
[isPhysioOnPath, pathPhysIO] = tapas_physio_check_path();

if ~isPhysioOnPath
    addpath(genpath(pathPhysIO));
    fprintf('added PhysIO path recursively: %s\n', pathPhysIO);
else
    fprintf('OK.\n');
end


%% Check and add SPM path
fprintf('Checking Matlab SPM path now...');
[isSpmOnPath, pathSpm] = tapas_physio_check_spm();
if ~isSpmOnPath
    
    if isempty(pathSpm) % we don't know where to look for, prompt user
        fprintf('\n\t');
        pathSpm = input('No SPM path found. If you want to add it, just enter it now [ENTER to continue]:', 's');
    end
    
    if exist(pathSpm, 'dir')
        addpath(pathSpm);
        fprintf('added SPM path: %s\n', pathSpm);
    else
        fprintf('No SPM path found. Skipping Batch Editor GUI integration.\n');
        fprintf('\nFinished!\n\n');
        return 
    end
    
else
    fprintf('OK.\n');
end


%% Check PhysIO/Matlab Integration via Batch Editor
fprintf('Checking whether PhysIO/code folder is a subfolder of SPM/toolbox (or a link within there)...');

isVerbose = false; % we will try to create link, don't warn yet
[isPhysioVisibleForSpmBatchEditor, pathSpm, pathPhysIO] = ...
    tapas_physio_check_spm_batch_editor_integration(isVerbose);

if ~isPhysioVisibleForSpmBatchEditor
    fprintf('No link found. Trying to create one...');
    cmdString = tapas_physio_create_spm_toolbox_link(pathPhysIO);
    
    % try again...
    [isPhysioVisibleForSpmBatchEditor, pathSpm, pathPhysIO] = ...
        tapas_physio_check_spm_batch_editor_integration();
    
    if isPhysioVisibleForSpmBatchEditor
        fprintf('OK.\n');
    else
        fprintf(['Failed to create link. You might need admin rights to run\n\t %s \n' ,...
            ' in your command window (cmd (Windows) or bash/terminal/shell (Unix/Mac)]\n'], ...
            cmdString);
    end
else
    fprintf('OK.\n');
end


%% Summary of checks
isPhysioCorrectlyInitialized = ~isempty(pathSpm) && ~isempty(pathPhysIO) && ...
    isPhysioVisibleForSpmBatchEditor;

if isPhysioCorrectlyInitialized
    disp('Success: PhysIO successfully installed, integration with Batch Editor possible.')
    fprintf('Updating SPM batch editor information...')
    spm_jobman('initcfg');
    fprintf('done.\n\n');
    fprintf('Finished!\n\n');
else
    if isempty(pathPhysIO)
        pathPhysIOHere = fileparts(mfilename('fullpath'));
        warning(sprintf(['\n PhysIO not setup correctly, add %s to the path, e.g., via ' ...
            '\n         addpath %s'],  pathPhysIOHere))
    else
        warning(sprintf(['\n PhysIO integration with SPM not setup. \n' ...
            ' PhysIO will run in Matlab, but not with SPM Batch Editor GUI. \n' ...
            ' Follow instructions above to link the paths for batch editor use.']))
    end
end
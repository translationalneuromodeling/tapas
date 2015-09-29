function args = report_contrasts_tapas_physio_check_efficacy(varargin)
% This function reports all relevant F-contrast-maps for physIO-created regressors
% for the specified subjects
%
% Input parameters are specified via name/value pairs, e.g.
%
%   args = report_contrasts_tapas_physio_check_efficacy(...
%               'fileReport', 'physio.ps', ...
%               'maskSubjects', 'LK3*', ...
%               'indReportContrasts', 2)
%
%   IN
%       defaults
%                      pathSPM: '~/Documents/code/matlab/spm12b'
%                   pathPhysIO: '~/Documents/code/matlab/spm12b/toolbox/PhysIO'
%                   fileReport: '~/Dropbox/Andreiuta/physio_rest_ioio_pharm/physio_IOIO_pharm/results/PhysIOTest_cardiac_overview_inferior_sliceParallel.ps'
%                 pathDataRoot: '/Users/kasperla/Dropbox/Andreiuta/physio_rest_ioio_pharm/physio_IOIO_pharm/glmAnalysis'
%                 maskSubjects: 'DMPAD_*'
%                       dirGLM: ''
%               maskStructural: '^meanfunct_rest.nii'
%           namesPhysContrasts: {5x1 cell}
%       indReportPhysContrasts: 2
%      reportContrastThreshold: 1.0000e-03
%     reportContrastCorrection: 'none'
%       reportContrastPosition: [0 0 -30]
%                fovMillimeter: 0
%          doPlotSliceParallel: 1
%                        model: [1x1 struct]

%       
%   OUT
%    args       structure of all output parameters
%
% Author: Lars Kasper
% Created: 2014-01-21
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TNU CheckPhysRETROICOR toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_check_efficacy.m 541 2014-10-11 16:40:49Z kasperla $

%% ========================================================================
% START #MOD


% general paths study
defaults.pathSPM         = '~/Documents/code/matlab/spm12b';
defaults.pathPhysIO      = '~/Documents/code/matlab/spm12b/toolbox/PhysIO';
defaults.fileReport      = ['~/Dropbox/Andreiuta/physio_rest_ioio_pharm/' ...
    'physio_IOIO_pharm/results/PhysIOTest_cardiac_overview_inferior_sliceParallel.ps']; % where contrast maps are saved

% root directory holding all subject directories
defaults.pathDataRoot    = '/Users/kasperla/Dropbox/Andreiuta/physio_rest_ioio_pharm/physio_IOIO_pharm/glmAnalysis';

% prefix of all subject directories
defaults.maskSubjects    = 'DMPAD_*';

% GLM analysis subdirectory of subject folder
defaults.dirGLM          = '';

% includes subdirectory of subject folder and file name mask
defaults.maskStructural  = '^meanfunct_rest.nii';

% names of physiological contrasts to be reported
% namesPhysContrasts = {
%             'All Phys Regressors'
%             'Cardiac Regressors'
%             'Respiratory Regressors'
%             'Cardiac X Respiratory Interaction'
%             'Movement Regressors'
%             };

defaults.namesPhysContrasts = {
    'All Phys'
    'Cardiac'
    'Respiratory'
    'Card X Resp Interation'
    'Movement'
    };

% selection of physiological contrasts to be reported, corresponding to
% namesPhysContrasts order
defaults.indReportPhysContrasts = 2;


defaults.reportContrastThreshold     = 0.001; % 0.05; 0.001;
defaults.reportContrastCorrection    = 'none'; % 'FWE'; 'none';
%reportContrastPosition      = [0 -15 -2*16]; 'max'; % 'max' to jump to max; or [x,y,z] in mm
%fovMillimeter               = 50; %mm; choose 0 to plot whole FOV (bounding box)
defaults.reportContrastPosition      = [0 0 -30]; 'max'; % 'max' to jump to max; or [x,y,z] in mm
defaults.fovMillimeter               = 0; %mm; choose 0 to plot whole FOV (bounding box)

% if true, voxel space (parallel to slices), not world space (with interpolation) is used
defaults.doPlotSliceParallel          = true; 

physio                                = tapas_physio_new('RETROICOR');
defaults.model                        = physio.model; % holding number of physiological regressors

% END #MOD
%% ========================================================================

% convert property/value pairs by updating defaults into variables
args = propval(varargin, defaults);
strip_fields(args);

scans = dir(fullfile(pathDataRoot,[maskSubjects '*']));
scans = {scans.name};
subjectIndices = 1:length(scans);

delete(fileReport);
addpath(pathPhysIO);
addpath(pathSPM);
spm('defaults', 'fMRI');
% spm_jobman('initcfg');

for s = subjectIndices
    
    try
        dirSubject = scans{s};
        pathSubject = fullfile(pathDataRoot,dirSubject); %dirSubject = scan
        pathAnalysis = fullfile(pathDataRoot,dirSubject,dirGLM);
        fileSPM = fullfile(pathAnalysis, 'SPM.mat');
        fileStruct = spm_select('ExtFPList', pathSubject, maskStructural);
        
        
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create and plot phys regressors F-contrasts
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if exist(fileSPM, 'file')
            tapas_physio_report_contrasts()
            else % no file, report that
            fprintf('no SPM.mat: %s\n', dirSubject);
        end
    catch
        warning(sprintf('Subject ID %d: %s did not run through', s, dirSubject));
    end
end
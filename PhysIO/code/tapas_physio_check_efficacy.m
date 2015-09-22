% This script reports all relevant F-contrast-maps for physIO-created regressors
% for the specified subjects
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
pathSPM         = '~/Documents/code/matlab/spm12b';
pathPhysIO      = '~/Documents/code/matlab/spm12b/toolbox/PhysIO';
fileReport      = ['~/Dropbox/Andreiuta/physio_rest_ioio_pharm/' ...
    'physio_IOIO_pharm/results/PhysIOTest_cardiac_overview_inferior_sliceParallel.ps']; % where contrast maps are saved

% logfile names sorted per session
nSess = 1;

% subject directories to be included into analysis

% root directory holding all subject directories
pathDataRoot    = '/Users/kasperla/Dropbox/Andreiuta/physio_rest_ioio_pharm/physio_IOIO_pharm/glmAnalysis';

% prefix of all subject directories
maskSubjects    = 'DMPAD_*';

% GLM analysis subdirectory of subject folder
dirGLM          = '';

% includes subdirectory of subject folder and file name mask
maskStructural  = '^meanfunct_rest.nii';

% names of physiological contrasts to be reported
% namesPhysContrasts = {
%             'All Phys Regressors'
%             'Cardiac Regressors'
%             'Respiratory Regressors'
%             'Cardiac X Respiratory Interaction'
%             'Movement Regressors'
%             };

namesPhysContrasts = {
    'All Phys'
    'Cardiac'
    'Respiratory'
    'Card X Resp Interation'
    'Movement'
    };

% selection of physiological contrasts to be reported, corresponding to
% namesPhysContrasts order
indReportPhysContrasts = 2;


reportContrastThreshold     = 0.001; % 0.05; 0.001;
reportContrastCorrection    = 'none'; % 'FWE'; 'none';
%reportContrastPosition      = [0 -15 -2*16]; 'max'; % 'max' to jump to max; or [x,y,z] in mm
%fovMillimeter               = 50; %mm; choose 0 to plot whole FOV (bounding box)
reportContrastPosition      = [0 0 -30]; 'max'; % 'max' to jump to max; or [x,y,z] in mm
fovMillimeter               = 0; %mm; choose 0 to plot whole FOV (bounding box)

% if true, voxel space (parallel to slices), not world space (with interpolation) is used
doPlotSliceParallel = true; 

physio          = tapas_physio_new('RETROICOR');
model           = physio.model; % holding number of physiological regressors

% END #MOD
%% ========================================================================

scans = dir(fullfile(pathDataRoot,maskSubjects));
scans = {scans.name};
subjectIndices = 1:length(scans);

delete(fileReport);
addpath(pathPhysIO);
addpath(pathSPM);
spm('defaults', 'fMRI');
spm_jobman('initcfg');

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
            load(fileSPM);
            nContrasts = numel(indReportPhysContrasts);
            
            
            % Check whether contrasts already exist in SPM.mat
            indContrasts = zeros(nContrasts,1);
            for c = 1:nContrasts
                iC = indReportPhysContrasts(c);
                indContrasts(c) = tapas_physio_check_get_xcon_index(SPM, ...
                    namesPhysContrasts{iC});
            end
            
            % Generate contrasts only if not already existing
            matlabbatch = tapas_physio_check_prepare_job_contrasts(fileSPM, ...
                model, SPM, indReportPhysContrasts, pathPhysIO);
            matlabbatch{1}.spm.stats.con.consess(find(indContrasts)) = [];
            if ~isempty(matlabbatch{1}.spm.stats.con.consess)
                spm_jobman('run', matlabbatch);
                load(fileSPM);
            end
            
            % report contrasts
            for c = 1:nContrasts
                iC = indReportPhysContrasts(c);
                indContrasts(c) = tapas_physio_check_get_xcon_index(SPM, ...
                    namesPhysContrasts{iC});
                load(fullfile(pathPhysIO, 'tapas_physio_check_job_report'));
                matlabbatch{1}.spm.stats.results.spmmat = cellstr(fileSPM);
                matlabbatch{1}.spm.stats.results.conspec.titlestr = [dirSubject ' - ' namesPhysContrasts{iC}];
                matlabbatch{1}.spm.stats.results.conspec.contrasts = indContrasts(c);
                
                % contrast report correction
                matlabbatch{1}.spm.stats.results.conspec.thresh = reportContrastThreshold;
                matlabbatch{1}.spm.stats.results.conspec.threshdesc = reportContrastCorrection;
                
                spm_jobman('run', matlabbatch);                     % report result
                %                     spm_print(fileReport)
                spm_sections(xSPM,hReg, fileStruct);                % overlay structural
                
                % voxel, not world space
                if doPlotSliceParallel
                    spm_orthviews('Space',1)
                end
                
                spm_orthviews('Zoom', fovMillimeter); % zoom to FOV*2 view
                spm_orthviews('Interp', 0); % next neighbour interpolation plot
               
                
                if isequal(reportContrastPosition, 'max');
                    spm_mip_ui('Jump',spm_mip_ui('FindMIPax'),'glmax'); % goto global max
                else
                    spm_mip_ui('SetCoords', reportContrastPosition, ...
                        spm_mip_ui('FindMIPax')); % goto global max
                end
                
                
                % spm_print always prepend current directory to print-file
                % name :-(
                [pathReport, filenameReport] = fileparts(fileReport);
                pathTmp = pwd;
                cd(pathReport);
                spm_print(filenameReport);
                cd(pathTmp);
            end
            
            titstr = [dirSubject, ' - SPM.xX.X'];
            title(regexprep(titstr,'_','\\_'));
            set(gcf,'Name', titstr);
            fprintf('good SPM: %s\n', dirSubject);
        else % no file, report that
            fprintf('no SPM.mat: %s\n', dirSubject);
        end
    catch
        warning(sprintf('Subject ID %d: %s did not run through', s, dirSubject));
    end
end
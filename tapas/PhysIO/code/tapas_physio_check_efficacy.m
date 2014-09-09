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
% $Id: tapas_physio_check_efficacy.m 416 2014-01-21 03:29:12Z kasperla $

%% ========================================================================
% START #MOD

% general paths study
pathSPM         = '~/code/matlab/spm12b';
pathPhysIO      = '~/code/matlab/smoothing_trunk/tSNR_fMRI_SPM/CheckPhysRETROICOR/PhysIOToolbox/code';
fileReport      = '~/PhysIOTest.ps'; % where contrast maps are saved

% logfile names sorted per session
nSess = 1;

% subject directories to be included into analysis
dirData         = '';
dirScans        = '/cluster/scratch_xl/shareholder/klaas/dandreea/IOIO/data';
maskScans       = 'F_*';
maskGLM         = '/signedPE';
maskStructural  = 'spm_pre/struct/1_2';

% END #MOD
%% ========================================================================

scans = dir(fullfile(dirScans,maskScans));
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
    pathSubject = fullfile(dirScans,dirSubject); %dirSubject = scan
    pathAnalysis = fullfile(dirScans,dirSubject,maskGLM);
    fileSPM = fullfile(pathAnalysis, 'SPM.mat');
    fileStruct = spm_select('FPList', fullfile(pathSubject, maskStructural), '^wm.*nii');
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create and plot phys regressors F-contrasts
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if exist(fileSPM, 'file')
        load(fileSPM);
        
        % Do the physiological contrasts already exist
        phys_cnames = {
            'All Phys Regressors'
            'Cardiac Regressors'
            'Respiratory Regressors'
            'Cardiac X Respiratory Interaction'
            'Movement Regressors'
            };
        indC = zeros(5,1);
        
        matlabbatch = tapas_physio_check_prepare_job_contrasts(fileSPM, ...
            SPM, pathPhysIO);
        
        % generate contrasts only if not already existing
        for indPhysCon = 1:5
            indC(indPhysCon) = tapas_physio_check_get_xcon_index(SPM, ...
                phys_cnames{indPhysCon});
        end
        matlabbatch{1}.spm.stats.con.consess(find(indC)) = [];
        if ~isempty(matlabbatch{1}.spm.stats.con.consess)
            spm_jobman('run', matlabbatch);
            load(fileSPM);
        end
        % report contrasts
        for indPhysCon = 1:5
            indC(indPhysCon) = tapas_physio_check_get_xcon_index(SPM, ...
                phys_cnames{indPhysCon});
            load(fullfile(pathPhysIO, 'tapas_physio_check_job_report'));
            matlabbatch{1}.spm.stats.results.spmmat = cellstr(fileSPM);
            matlabbatch{1}.spm.stats.results.conspec.titlestr = [dirSubject ' - ' phys_cnames{indPhysCon}];
            matlabbatch{1}.spm.stats.results.conspec.contrasts = indC(indPhysCon);
            spm_jobman('run', matlabbatch);                     % report result
            %                     spm_print(fileReport)
            spm_sections(xSPM,hReg, fileStruct);                % overlay structural
            spm_mip_ui('Jump',spm_mip_ui('FindMIPax'),'glmax'); % goto global max
            spm_print(fileReport)
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
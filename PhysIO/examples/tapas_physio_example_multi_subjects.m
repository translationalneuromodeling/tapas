% Script tapas_physio_example_multi_subjects(input)
% Shows how to generalize single-subject batch for all subjects 
%
%
%   See also tapas_physio_new

% Author: Lars Kasper
% Created: 2015-07-31
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% # MOD - Change parameters to your needs in this section %%%%%%%%%%%%%%%%

% set 1 to look at batch, to 0 for running it for all subjects
doReviewBatch       = false; 

pathStudy           = '/Users/kasperla/studies/physiotest';

dirSubjectArray     = {
    'subject10'
    'subject11'
    };

pathStudy = '/Users/kasperla/Documents/code/matlab/smoothing_trunk/PhysIOToolbox/examples/Philips';
dirSubjectArray = {
    'ECG3T'
    'ECG3T'
    };

% folder, where multiple_regressors, physio.mat and output figures are saved,
% typically analysis folder of the subject
dirOutput           = 'glm'; 

filePhysCardiac     = 'SCANPHYSLOG.log';
filePhysRespiratory = 'SCANPHYSLOG.log';
fileRealignmentPar  = 'rp_fmri.txt';
jobfile             = 'tapas_physio_example_spm_job_ECG3T.m';



%% # END MOD Loop over subjects, filling in subject-specific data

nSubjects = numel(dirSubjectArray);
spm('defaults', 'FMRI');

for iSubject = 1:nSubjects
    
    % load template matlabbatch
    clear matlabbatch
    run(jobfile);
    physio = matlabbatch{1}.spm.tools.physio;
   
    % construct subject-specific data
    pathSubject = fullfile(pathStudy, dirSubjectArray{nSubjects});
    fullpathFilePhysCardiac = fullfile(pathSubject, filePhysCardiac);
    fullpathFilePhysRespiratory = fullfile(pathSubject, filePhysRespiratory);
    fullpathFileRealignment = fullfile(pathSubject, fileRealignmentPar);
    
    % overwrite matlabbatch structure parameters
    physio.save_dir = {pathSubject};
    physio.log_files.cardiac = {fullpathFilePhysCardiac};
    physio.log_files.respiration = {fullpathFilePhysRespiratory};
    physio.log_files.scan_timing = {};
    
    physio.model.movement.yes.file_realignment_parameters = ...
        {fullpathFileRealignment};
    
    % run/examine job for this subject
    matlabbatch{1}.spm.tools.physio = physio;
    
    if doReviewBatch
        spm_jobman('interactive', matlabbatch);
    else
        spm_jobman('run', matlabbatch);
    end
end

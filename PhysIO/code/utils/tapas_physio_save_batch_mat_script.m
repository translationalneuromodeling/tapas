function physio = tapas_physio_save_batch_mat_script(fileBatchM, pathOutput, ...
    doRemoveEmptyDefaults)
% Saves .m-matlabbatch-file as .mat and as spm-independent matlab-script
%
%   physio = tapas_physio_save_batch_mat_script(fileBatchM)
%
% IN
%   fileBatchM      either .m-file of batch job or matlabbatch structure
%                   variable
%   pathOutput      path where new .mat and matlab-script file are saved
%                   default: same as fileBatchM
%   doRemoveEmptyDefaults
%                   If true, matlab batch lines with [] '' {} are not
%                   printed in output .m files
%                   default: true
%
% OUT
%   physio          physio-structure, see also tapas_physio_new
%   
% SIDE EFFECTS
%   <fileBatch>.mat         created, is .mat-job for spm_jobman, holding a
%                           variable matlabbatch
%   <fileBatch>_matlab_script.m    created, is standalone (w/o spm) matlab script version
%
% EXAMPLE
%   tapas_physio_save_batch_mat_script('example_main_job_ECG7T.m');
%
%
%   See also cfg_util gencode

% Author: Lars Kasper
% Created: 2015-07-19
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 1
    fileBatchM = 'example_spm_job_ECG7T.m';
end

% file given run it!
doSaveBatchM = true;
if ~ischar(fileBatchM)
    matlabbatch = fileBatchM;
    fileBatchM = fullfile(pwd, 'physio_job.m');
    fileBatchMat = fullfile(pwd, 'physio_job.mat');
else
    [fp,fn,ext] = fileparts(fileBatchM);
    switch ext
        case '.mat' % indeed a .mat file
            fileBatchMat = fileBatchM;
            fileBatchM = fullfile(fp,[fn '.m']);
            load(fileBatchMat);
        case '.m'
            doSaveBatchM = false; % does exist already
            fileBatchMat = fullfile(fp,[fn '.mat']);
            run(fileBatchM);
    end
end

fileScript = regexprep(fileBatchM, {'spm_job', 'job'}, 'matlab_script');

if isequal(fileScript,fileBatchM)
    fileScript = regexprep(fileBatchM, '\.m', '_matlab_script\.m');
end

if nargin >=2
    %% replace paths in output file
    pathBatchM =  fileparts(fileBatchM);
    if isempty(pathBatchM)
        % no path originally
        fileBatchMOut = fullfile(pathOutput, fileBatchM);
        fileBatchMat = fullfile(pathOutput, fileBatchMat);
        fileScript = fullfile(pathOutput, fileScript);
    else
        fileBatchMOut = regexprep(fileBatchM, pathBatchM, pathOutput);
        fileBatchMat = regexprep(fileBatchMat, pathBatchM, pathOutput);
        fileScript = regexprep(fileScript, pathBatchM, pathOutput);
    end
    [~,~] = mkdir(pathOutput);
end

if nargin < 3
    doRemoveEmptyDefaults = true;
end

if ~exist('cfg_files', 'file')
    spm_jobman('initcfg');
end
spm('defaults', 'FMRI');

% set up matlabbatch as job
jobId = cfg_util('initjob', matlabbatch);
fileTemp = sprintf('tmp_%s', datestr(now, 'yymmdd_HHMMSS'));

% write out job
cfg_util('genscript', jobId, pwd, fileTemp);

clear matlabbatch

% as input batch, but with filled out defaults

fileTempJob = [fileTemp '_job.m'];
run(fileTempJob);

% delete temporary files

if doSaveBatchM
    delete([fileTemp '.m']);
    movefile(fileTempJob, fileBatchMOut);
else
    delete([fileTemp '.m'], fileTempJob);
end

% remove newly introduced absolute paths :-(
pathJob = fileparts(fileBatchM);

% also pathNow, since cfg-stuff adds the paths of the directory it was executed
% in :-(
pathNow = pwd;
matlabbatch{1}.spm.tools.physio = tapas_physio_replace_absolute_paths(...
    matlabbatch{1}.spm.tools.physio, {pathJob, pathNow});


% save matlabbatch to mat-file

save(fileBatchMat, 'matlabbatch')

% convert to script variable and save to file
physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);

% write out variable strings and remove lines that set empty values

str = gencode(physio)';

if doRemoveEmptyDefaults
    indLineRemove = tapas_physio_find_string(str, ...
        {'= \[\];', '= {};', '= {''''};', '= '''';'});
    indLineRemove = cell2mat(indLineRemove);
    str(indLineRemove) = [];
end

% add comments to write and generating line for physio-structure, 
% and save to matlab_script-file
str = [
    {'%% Example script using PhysIO with Matlab only (no SPM needed)'}
    {'%  For documentation of the parameters, see also tapas_physio_new (e.g., via edit tapas_physio_new)'}
    {''}
    {'%% Create default parameter structure with all fields'}
    {'physio = tapas_physio_new();'}
    {''}
    {'%% Individual Parameter settings. Modify to your need and remove default settings'}
    str
    {''};
    {'%% Run physiological recording preprocessing and noise modeling'}
    {'physio = tapas_physio_main_create_regressors(physio);'}
    ];

nLines = numel(str);
fid = fopen(fileScript, 'w+');

for iLine = 1:nLines
    fprintf(fid, '%s\n', str{iLine});
end
fclose(fid);
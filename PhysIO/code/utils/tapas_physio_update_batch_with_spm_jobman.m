function matlabbatch = tapas_physio_update_batch_with_spm_jobman(matlabbatch, ...
    fileBatchM)
%  loads existing matlabbatch into SPM Job manager and returns updated
%  matlabbatch object (e.g., if created with old version and new fields
%  were added to matlabbatch structure)
%
%    matlabbatch = tapas_physio_update_batch_with_spm_jobman(matlabbatch)
%
% IN
%   matlabbatch     matlabbatch for PhysIO, e.g. from loading `.mat` or 
%                   `run(fileBatchM)`
%   fileBatchM      [optional] 
%                   file name of batch .m that matlabbatch came from
%                   (proxy to retrieve absolute file paths from matlabbatch
%                   default: ''
% OUT
%
% EXAMPLE
%   tapas_physio_update_batch_with_spm_jobman
%
%   See also tapas_physio_save_batch_mat_script
 
% Author:   Lars Kasper
% Created:  2022-03-30
% Copyright (C) 2022 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
if nargin < 2
    fileBatchM = '';
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

delete([fileTemp '.m'], fileTempJob);

% remove newly introduced absolute paths from spm job manager :-(
pathJob = fileparts(fileBatchM);
if isempty(pathJob)
    pathJob = '';
end

% also pathNow, since cfg-stuff adds the paths of the directory it was executed
% in :-(
pathNow = pwd;
matlabbatch{1}.spm.tools.physio = tapas_physio_replace_absolute_paths(...
    matlabbatch{1}.spm.tools.physio, {pathJob, pathNow});

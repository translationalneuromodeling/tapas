function physio = tapas_physio_update_from_job(physio, job, ...
    jobPropertyArray, physioPropertyArray, isBranch, branchProperty)
% Updates properties of physio-structure from a job-structure
%
%   physio = tapas_physio_update_from_job(physio, job, ...
%    jobPropertyArray, physioPropertyArray, isBranch)
%
% IN
%   physio      physio-structure. See also tapas_physio_new
%   job         spm_jobman job, as created by See also
%               tapas_physio_cfg_matlabbatch
%   jobPropertyArray  cell(nProperties,1)
%               (sub-)properties of job-input that shall update
%               physio-structure
%               e.g. thresh.cardiac.posthoc_cpulse_select
%   physioPropertyArray cell(nProperties,1)
%               sub-properties of physio-structure that shall be updated by
%               job
%               e.g. thresh.cardiac.posthoc_cpulse_select
%   isBranch    true or false or cell(nProperties,1) of true/false
%               If true, a branch is assumed as job-property, and the field
%               name of the job-branch will be used as a separate
%               (sub-) property of physio
%   branchProperty string or cell(nProperties,1) of strings that
%               jobProperty-field name itself shall be transferred to
%               default: 'method'
%
% OUT
%   physio      physio-structure with updated properties according to job
% EXAMPLE
%   physio = tapas_physio_update_from_job(physio, job, ...
%    'thresh.cardiac.posthoc_cpulse_select', ...
%       'thresh.cardiac.posthoc_cpulse_select, true, 'method')
%
%   will set all properties set in
%   job.thresh.cardiac.posthoc_cpulse_select.manual (or load or off)
%   also in physio.thresh.cardiac.posthoc_cpulse_select and will also set
%   physio.thresh.cardiac.posthoc_cpulse_select.method = 'manual' (or load
%   or off
%
%   physio = tapas_physio_update_from_job(physio, job, ...
%    'thresh.cardiac.posthoc_cpulse_select', ...
%       'thresh.cardiac.posthoc_cpulse_select, false)
%
%   will just take all existing fields of job.thresh.cardiac.posthoc_cpulse_select
%   and overwrite with this the corresponding fields of
%   physio.thresh.cardiac.posthoc_cpulse_select
%
%   See also

% Author: Lars Kasper
% Created: 2015-01-05
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



jobPropertyArray    = cellstr(jobPropertyArray);
physioPropertyArray = cellstr(physioPropertyArray);

if nargin < 5
    isBranch = true;
end

if nargin < 6
    branchProperty = 'method';
end

nProperties = numel(jobPropertyArray);


if ~iscell(isBranch)
    isBranch = repmat({isBranch}, nProperties,1);
end

if ~iscell(branchProperty)
    branchProperty = repmat({branchProperty}, nProperties,1);
end


for p = 1:nProperties
    try
        currentProperty = eval(sprintf('job.%s', jobPropertyArray{p}));
    catch err
        currentProperty = [];
        tapas_physio_log(sprintf('No property %s defined in job (error: %s)', ...
            jobPropertyArray{p}, err.message), [], 1);
    end
    
    if ~isempty(currentProperty) % no error in retrieval => continue parsing!
        
        % overwrite properties of physio with sub-properties of job, also
        % set value of branchProperty to name of current property
        if isBranch{p}
            valueBranchArray = fields(currentProperty);
            valueBranch = valueBranchArray{1};
            eval(sprintf('physio.%s.%s = valueBranch;', ...
                physioPropertyArray{p}, branchProperty{p}));
            
            currentProperty = currentProperty.(valueBranch);
        end
        
        
        % update property itself, if it has no sub-properties
        if ~isstruct(currentProperty)
            eval(sprintf('physio.%s = currentProperty;', ...
                physioPropertyArray{p}));
        else
            
            % update all existing sub-properties in job to physio
            
            subPropArray = fields(currentProperty);
            nFields = numel(subPropArray);
            
            for f = 1:nFields
                eval(sprintf('physio.%s.%s = currentProperty.%s;', ...
                    physioPropertyArray{p},subPropArray{f}, subPropArray{f}));
            end
        end
    end
end
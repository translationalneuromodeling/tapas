function [SPM, matlabbatch, indContrastsExisting, indContrastsCreate] = ...
    tapas_physio_create_missing_physio_contrasts(SPM, model, namesPhysContrasts)
% Creates all valid physiological subset regressors for given PhysIO model
% that are not already existing in SPM, ignores
%
% [matlabbatch, indContrastsExisting, indContrastsCreate] = ...
%     tapas_physio_create_missing_physio_contrasts(SPM, model, namesPhysContrasts)
%
% IN
%   SPM     SPM structured variable
%   model  physio.model
%   namesPhysContrasts 
%           optional cell of strings determining which contrasts shall be
%           listed for checking and, if needed, creation
%           NOTE: invalid names will be ignored
%           default: all phys contrasts will be queried for
%           creation, as listed in 
%
% OUT
%   SPM     SPM structured variable with updated contrasts
%   matlabbatch 
%           that was used to create new valid contrasts
%   indContrastsExisting
%           indices of namesPhysContrasts that existed in design matrix
%   indContrastsCreate
%           indices of namesPhysContrasts that had to be created
%
% EXAMPLE
%   tapas_physio_create_missing_physio_contrasts
%
%   See also tapas_physio_check_prepare_job_contrasts
%   See also tapas_physio_report_contrasts

% Author: Lars Kasper
% Created: 2016-10-03
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 3
    namesPhysContrasts = tapas_physio_get_contrast_names_default();
end

nContrasts = numel(namesPhysContrasts);
fileSpm = fullfile(SPM.swd, 'SPM.mat');

%% Check whether contrasts already exist in SPM.mat
indContrasts = zeros(nContrasts,1);
for c = 1:nContrasts
    indContrasts(c) = tapas_physio_check_get_xcon_index(SPM, ...
        namesPhysContrasts{c});
end


%% Generate contrasts only if not already existing

indContrastsExisting = find(indContrasts);

if ~isempty(model)
    indContrastsCreate      = find(~indContrasts);
    namesContrastsCreate    = namesPhysContrasts(indContrastsCreate);
    matlabbatch             = tapas_physio_check_prepare_job_contrasts(fileSpm, ...
        model, SPM, indContrastsCreate, namesContrastsCreate);
    if ~isempty(matlabbatch{1}.spm.stats.con.consess)
        spm_jobman('run', matlabbatch);
    end
else
    error('No physio.model specified');
end

if nargout
    load(fileSpm);
end
function [matlabbatch, indValidContrasts] = tapas_physio_check_prepare_job_contrasts(fileSPM, ...
    model, SPM, indReportPhysContrasts, namesPhysContrasts)
% adapts contrast generator for physiogical regressors to actual SPM-file
% (directory & design matrix columns)
%
%   
% IN
%   fileSpm
%   model
%   SPM
%   indReportPhysContrasts  [1, nContrasts] vector of phys contrast ids to be
%                           reported, in their default order
%   indValidContrasts       contrasts ids that could be queried with existing
%                           physIO model
%   namesPhysContrasts      if contrasts shall be named differently,
%                           entered here
% Author: Lars Kasper
% Created: 2014-01-21
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 6
    namesPhysContrasts = tapas_physio_get_contrast_names_default();
end

if ~exist('SPM', 'var'), load(fileSPM); end

[colPhys, colCard, colResp, colMult, colHRV, colRVT, colRois, colMove, colAll] = ...
    tapas_physio_check_get_regressor_columns(SPM, model);
con{1} = colPhys;
con{2} = colCard;
con{3} = colResp;
con{4} = colMult;
con{5} = colHRV;
con{6} = colRVT;
con{7} = colRois;
con{8} = colMove;
con{9} = colAll;

pathCodePhysIO = fileparts(mfilename('fullpath'));
load(fullfile(pathCodePhysIO,'tapas_physio_check_job_contrasts.mat'));
matlabbatch{1}.spm.stats.con.spmmat{1} = fileSPM;

% execute computation only for non-empty contrasts;
indValidContrasts = intersect(indReportPhysContrasts, ...
    find(~cellfun(@isempty, con)));
nContrasts = numel(indValidContrasts);
for c = 1:nContrasts
    iC = indValidContrasts(c);
    F{c} = zeros(max(con{iC}));
    F{c}(sub2ind(size(F{c}),con{iC}, con{iC})) = 1;
    matlabbatch{1}.spm.stats.con.consess{c}.fcon.convec{1} = F{c};
    
    matlabbatch{1}.spm.stats.con.consess{c}.fcon.name = ...
            namesPhysContrasts{indValidContrasts(c)};
end

% remove additional contrasts in matlabbatch
matlabbatch{1}.spm.stats.con.consess(nContrasts+1:end) = [];
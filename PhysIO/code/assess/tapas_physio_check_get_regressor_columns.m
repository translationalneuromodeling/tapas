function [colPhys, colCard, colResp, colMult, colHRV, colRVT, colRois, colMove, colAll] = ...
    tapas_physio_check_get_regressor_columns(SPM, model)
%
% returns indices of physiological regressors in an SPM design-matrix,
% pooled over all sessions for later creation of an F-contrast
%
% INPUT:
%   SPM     SPM.mat
%   model   physIO.model-structure
%
%
% OUTPUT:
%   colPhys     - index vector of all physiological regressor columns in design matrix (from SPM.xX.names)
%   colCard     - index vector of cardiac regressor columns in design matrix (from SPM.xX.names)
%   colResp     - index vector of respiratory regressor columns in design matrix (from SPM.xX.names)
%   colMult     - index vector of interaction cardiac X respiration regressor columns in design matrix (from SPM.xX.names)
%   colHRV      - index vector of heart rate variability column in design matrix (from SPM.xX.names)
%   colRVT      - index vector of respiratory volume per time column in design matrix (from SPM.xX.names)
%   colRois     - index vector of noise rois regressors (from SPM.xX.names)
%   colMove     - index vector of movement regressor columns in design matrix (from SPM.xX.names)
%   colAll      - colPhys and colMove (all nuisance regressors!)

% Author: Lars Kasper
% Created: 2014-01-21
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TNU CheckPhysRETROICOR toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


nSess = length(SPM.Sess);

iCard = 1;

if nargin < 2
    nMove = 6;
    nCard = 6;
    nResp = 8;
    nMult = 4;
    nHRV  = 0;
    nRVT  = 0;
    nRois = 0;
else
    
    if model.retroicor.include
                
        nCard = ~isempty(model.retroicor.order.c);
        if nCard
            nCard = model.retroicor.order.c*2;
        end
        
        nResp = ~isempty(model.retroicor.order.r);
        if nResp
            nResp = model.retroicor.order.r*2;
        end
        
        nMult = ~isempty(model.retroicor.order.cr);
        if nMult
            nMult = model.retroicor.order.cr*4;
        end
        
        
    else
        nCard = 0;
        nResp = 0;
        nMult = 0;
    end
    
    
    % only if following models were calculated
    nRois = model.noise_rois.include.*sum(model.noise_rois.n_components);
    nMove = model.movement.include.*model.movement.order;
    nHRV  = model.hrv.include.*numel(model.hrv.delays);
    nRVT  = model.rvt.include.*numel(model.rvt.delays);
    
    assert(~model.other.include, 'SPM review not compatible with ''model.other''');
    
end

cnames = SPM.xX.name';

colCard = [];
colResp  = [];
colMult = [];
colRois = [];
colMove = [];
colHRV  = [];
colRVT  = [];
for s = 1:nSess
    
    cname = ['Sn(' int2str(s) ') R' int2str(iCard)];
    indC = find(strcmp(cnames, cname));
    if isempty(indC)
        % Check if SPM has loaded names directly
        indC = find(contains(cnames, model.R_column_names{1}, 'IgnoreCase', true), 1);
    end
    
    colCard = [colCard, indC:(indC+nCard - 1)];
    colResp = [colResp, (indC+nCard):(indC+nCard+nResp - 1)];
    colMult = [colMult, (indC+nCard+nResp):(indC+nCard+nResp+nMult - 1)];
    colHRV  = [colHRV, (indC+nCard+nResp+nMult):...
        (indC+nCard+nResp+nMult+nHRV-1)];
    colRVT  = [colRVT, (indC+nCard+nResp+nMult+nHRV):...
        (indC+nCard+nResp+nMult+nHRV+nRVT-1)];
    colRois = [colRois, (indC+nCard+nResp+nMult+nHRV+nRVT):...
        (indC+nCard+nResp+nMult+nHRV+nRVT+nRois-1)];
    colMove = [colMove, (indC+nCard+nResp+nMult+nHRV+nRVT+nRois):...
        (indC+nCard+nResp+nMult+nHRV+nRVT+nRois+nMove-1)];
end

colPhys = [colCard colResp colMult colHRV colRVT colRois];
colAll  = [colPhys colMove];
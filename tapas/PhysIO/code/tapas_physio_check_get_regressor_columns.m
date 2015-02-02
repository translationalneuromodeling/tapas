function [colAll, colCard, colResp, colMult, colHRV, colRVT, colMove] = ...
    tapas_physio_check_get_regressor_columns(SPM, model)
%
% returns indices of physiological regressors in an SPM design-matrix,
% pooled over all sessions for later creation of an F-contrast
%
% INPUT:
%   SPM     SPM.mat
%   model  physIO.model-structure
%    
%
% OUTPUT:
%   colAll      - index vector of all physiological regressor columns in design matrix (from SPM.xX.names)
%   colCard     - index vector of cardiac regressor columns in design matrix (from SPM.xX.names)
%   colResp     - index vector of respiratory regressor columns in design matrix (from SPM.xX.names)
%   colMult     - index vector of interaction cardiac X respiration regressor columns in design matrix (from SPM.xX.names)
%   colHRV      - index vector of heart rate variability column in design matrix (from SPM.xX.names)
%   colRVT      - index vector of respiratory volume per time column in design matrix (from SPM.xX.names)
%   colMove     - index vector of movement regressor columns in design matrix (from SPM.xX.names)
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
% $Id: tapas_physio_check_get_regressor_columns.m 581 2014-11-09 01:05:46Z kasperla $

nSess = length(SPM.Sess);

iCard = 1;

if nargin < 2
    nMove = 6;
    nCard = 6;
    nResp = 8;
    nMult = 4;
    nHRV  = 0;
    nRVT  = 0;
    
else
    
    if ~isempty(model.input_other_multiple_regressors)
        nMove = 6;
    else
        nMove = 0;
    end
    
    nCard = model.order.c*2;
    nResp = model.order.r*2;
    nMult = model.order.cr*4;
    
    % only for models with HRV or RVT, add these regressors
    nHRV  = any(strfind(upper(model.type), 'HRV'));
    nRVT  = any(strfind(upper(model.type), 'RVT'));
    
end

cnames = SPM.xX.name';

colCard = [];
colResp  = [];
colMult = [];
colAll  = [];
colMove = [];
colHRV  = [];
colRVT  = [];
for s = 1:nSess
    
    cname = ['Sn(' int2str(s) ') R' int2str(iCard)];
    indC = find(strcmp(cnames, cname));
    
    colCard = [colCard, indC:(indC+nCard - 1)];
    colResp = [colResp, (indC+nCard):(indC+nCard+nResp - 1)];
    colMult = [colMult, (indC+nCard+nResp):(indC+nCard+nResp+nMult - 1)];
    colHRV  = [colHRV, (indC+nCard+nResp+nMult):...
        (indC+nCard+nResp+nMult+nHRV-1)];
    colRVT  = [colRVT, (indC+nCard+nResp+nMult+nHRV):...
        (indC+nCard+nResp+nMult+nHRV+nRVT-1)];
    colMove = [colMove, (indC+nCard+nResp+nMult+nHRV+nRVT):...
        (indC+nCard+nResp+nMult+nHRV+nRVT+nMove-1)];
end

colAll = [colCard colResp colMult colHRV colRVT];
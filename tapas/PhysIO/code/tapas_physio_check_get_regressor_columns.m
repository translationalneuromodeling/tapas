function [colAll, colCard, colResp, colMult, colMove] = ...
    tapas_physio_check_get_regressor_columns(SPM, physio)
%
% returns indices of physiological regressors in an SPM design-matrix,
% pooled over all sessions for later creation of an F-contrast
%
% INPUT:
%   SPM     SPM.mat
%   physio  physIO-structure
%    .model
%
% OUTPUT:
%   colAll     - index vector of all physiological regressor columns in design matrix (from SPM.xX.names)
%   colCard     - index vector of cardiac regressor columns in design matrix (from SPM.xX.names)
%   colResp     - index vector of respiratory regressor columns in design matrix (from SPM.xX.names)
%   colMult     - index vector of interaction cardiac X respiration regressor columns in design matrix (from SPM.xX.names)
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
% $Id: tapas_physio_check_get_regressor_columns.m 415 2014-01-21 03:16:11Z kasperla $

nSess = length(SPM.Sess);

iCard = 1;

if nargin < 2
    nMove = 6;
    nCard = 6;
    nResp = 8;
    nMult = 4;
else
    model = physio.model;
    
    if ~isempty(model.input_other_multiple_regressors)
        nMove = 6;
    else
        nMove = 0;
    end
    
    nCard = model.c*2;
    nResp = model.r*2;
    nMult = model.cr*4;
end

cnames = SPM.xX.name';

colCard = [];
colResp  = [];
colMult = [];
colAll  = [];
colMove = [];
for s = 1:nSess
    
    cname = ['Sn(' int2str(s) ') R' int2str(iCard)];
    indC = find(cell2mat(cellfun(@(x) ~isempty(x), strfind(cnames, cname), 'UniformOutput', false)));
    
    colCard = [colCard, indC:(indC+nCard - 1)];
    colResp = [colResp, (indC+nCard):(indC+nCard+nResp - 1)];
    colMult = [colMult, (indC+nCard+nResp):(indC+nCard+nResp+nMult - 1)];
    colMove = [colMove, (indC+nCard+nResp+nMult):(indC+nCard+nResp+nMult+nMove-1)];
    
end

colAll = [colCard colResp colMult];
function matlabbatch = tapas_physio_check_prepare_job_contrasts(fileSPM, SPM, dirCheckPhys)
% adapts contrast generator for physiogical regressors to actual SPM-file
% (directory & design matrix columns)
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
% $Id: tapas_physio_check_prepare_job_contrasts.m 415 2014-01-21 03:16:11Z kasperla $

if ~exist('SPM', 'var'), load(fileSPM); end

[colAll, colCard, colResp, colMult, colMove] = ...
    tapas_physio_check_get_regressor_columns(SPM);
con{1} = colAll;
con{2} = colCard;
con{3} = colResp;
con{4} = colMult;
con{5} = colMove;

load(fullfile(dirCheckPhys,'tapas_physio_check_job_contrasts.mat'));
matlabbatch{1}.spm.stats.con.spmmat{1} = fileSPM;

for c = 1:5
    F{c} = zeros(max(con{c}));
    F{c}(sub2ind(size(F{c}),con{c}, con{c})) = 1;
    matlabbatch{1}.spm.stats.con.consess{c}.fcon.convec{1} = F{c};
end
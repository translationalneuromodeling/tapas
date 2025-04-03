function [pvec, pstruct] = tapas_ehgf_binary_pu_tbt_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Anna Yurova, Irene Sophia Plank, LMU University Hospital; 
% ehgf adaptation based on the function tapas_hgf_binary_pu_tbt_transp by Rebecca Lawson, Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pvec    = NaN(1,length(ptrans));
pstruct = struct;

l = r.c_prc.n_levels;

pvec(1:l)         = ptrans(1:l);                           % mu_0
pstruct.mu_0      = pvec(1:l);
pvec(l+1:2*l)     = exp(ptrans(l+1:2*l));                  % sa_0
pstruct.sa_0      = pvec(l+1:2*l);
pvec(2*l+1:3*l)   = ptrans(2*l+1:3*l);                     % rho
pstruct.rho       = pvec(2*l+1:3*l);
pvec(3*l+1:4*l-1) = exp(ptrans(3*l+1:4*l-1));              % ka
pstruct.ka        = pvec(3*l+1:4*l-1);
pvec(4*l:5*l-1)   = ptrans(4*l:5*l-1);                     % om
pstruct.om        = pvec(4*l:5*l-1);
pvec(5*l)         = ptrans(5*l);                           % eta0
pstruct.eta0      = pvec(5*l);
pvec(5*l+1)       = ptrans(5*l+1);                         % eta1
pstruct.eta1      = pvec(5*l+1);

return;

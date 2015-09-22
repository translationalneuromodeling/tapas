function [pvec, pstruct] = tapas_hgf_ar1_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
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
pvec(2*l+1:3*l)   = tapas_sgm(ptrans(2*l+1:3*l),1);        % phi
pstruct.phi       = pvec(2*l+1:3*l);
pvec(3*l+1:4*l)   = ptrans(3*l+1:4*l);                     % m
pstruct.m         = pvec(3*l+1:4*l);
pvec(4*l+1:5*l-1) = exp(ptrans(4*l+1:5*l-1));              % ka
pstruct.ka        = pvec(4*l+1:5*l-1);
pvec(5*l:6*l-1)   = ptrans(5*l:6*l-1);                     % om
pstruct.om        = pvec(5*l:6*l-1);
pvec(6*l)         = exp(ptrans(6*l));                      % al
pstruct.al        = pvec(6*l);

return;

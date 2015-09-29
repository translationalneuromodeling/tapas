function [pvec, pstruct] = tapas_hgf_whichworld_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

nw = r.c_prc.nw;

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1:nw)       = ptrans(1:nw);                     % mu2_0
pstruct.mu2_0    = pvec(1:nw);
pvec(nw+1:2*nw)  = exp(ptrans(nw+1:2*nw));           % sa2_0
pstruct.sa2_0    = pvec(nw+1:2*nw);
pvec(2*nw+1)     = ptrans(2*nw+1);                   % mu3_0
pstruct.mu3_0    = pvec(2*nw+1);
pvec(2*nw+2)     = exp(ptrans(2*nw+2));              % sa3_0
pstruct.sa3_0    = pvec(2*nw+2);
pvec(2*nw+3)     = tapas_sgm(ptrans(2*nw+3),r.c_prc.kaub); % ka
pstruct.ka       = pvec(2*nw+3);
pvec(2*nw+4)     = ptrans(2*nw+4);                   % om
pstruct.om       = pvec(2*nw+4);
pvec(2*nw+5)     = tapas_sgm(ptrans(2*nw+5),r.c_prc.thub); % th
pstruct.th       = pvec(2*nw+5);
pvec(2*nw+6)     = ptrans(2*nw+6);                   % m
pstruct.m        = pvec(2*nw+6);
pvec(2*nw+7)     = tapas_sgm(ptrans(2*nw+7),1); % phi
pstruct.phi      = pvec(2*nw+7);

return;
function [pvec, pstruct] = tapas_hgf_binary3l_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1)       = ptrans(1);                   % mu2_0
pstruct.mu2_0 = pvec(1);
pvec(2)       = exp(ptrans(2));              % sa2_0
pstruct.sa2_0 = pvec(2);
pvec(3)       = ptrans(3);                   % mu3_0
pstruct.mu3_0 = pvec(3);
pvec(4)     = exp(ptrans(4));                % sa3_0
pstruct.sa3_0 = pvec(4);
pvec(5)       = tapas_sgm(ptrans(5),r.c_prc.kaub); % ka
pstruct.ka    = pvec(5);
pvec(6)       = ptrans(6);                   % om
pstruct.om    = pvec(6);
pvec(7)       = tapas_sgm(ptrans(7),r.c_prc.thub); % th
pstruct.th    = pvec(7);

return;
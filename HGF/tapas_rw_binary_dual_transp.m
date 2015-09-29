function [pvec, pstruct] = tapas_rw_binary_dual_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1:2)     = tapas_sgm(ptrans(1:2),1); % v_0
pstruct.v_0   = pvec(1:2);
pvec(3)       = tapas_sgm(ptrans(3),1);   % alpha
pstruct.al    = pvec(3);
pvec(4)       = tapas_sgm(ptrans(4),1);   % kappa
pstruct.ka    = pvec(4);

return;
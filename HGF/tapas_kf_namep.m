function pstruct = tapas_kf_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pstruct = struct;

pstruct.g_0       = pvec(1);
pstruct.mu_0      = pvec(2);
pstruct.om        = pvec(3);
pstruct.pi_u      = pvec(4);

return;

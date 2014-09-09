function pstruct = tapas_hgf_binary3l_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pstruct = struct;

pstruct.mu2_0 = pvec(1);
pstruct.sa2_0 = pvec(2);
pstruct.mu3_0 = pvec(3);
pstruct.sa3_0 = pvec(4);
pstruct.ka    = pvec(5);
pstruct.om    = pvec(6);
pstruct.th    = pvec(7);

return;
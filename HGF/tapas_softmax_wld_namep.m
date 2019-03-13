function pstruct = tapas_softmax_wld_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2019 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pstruct = struct;

pstruct.be    = pvec(1);
pstruct.la_wd = pvec(2);
pstruct.la_ld = pvec(3);

return;

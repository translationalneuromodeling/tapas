function pstruct = tapas_hgf_categorical_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

no = 3;

pstruct = struct;

pstruct.mu2_0 = pvec(1:no);
pstruct.sa2_0 = pvec(no+1:2*no);
pstruct.mu3_0 = pvec(2*no+1);
pstruct.sa3_0 = pvec(2*no+2);
pstruct.ka    = pvec(2*no+3);
pstruct.om    = pvec(2*no+4);
pstruct.th    = pvec(2*no+5);

return;

function pstruct = tapas_hgf_jget_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2017 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pstruct = struct;

try
    l = r.c_prc.n_levels;
catch
    error('tapas:hgf:UndetNumLevels', 'Cannot determine number of levels');
end

pstruct.mux_0     = pvec(1:l);
pstruct.sax_0     = pvec(l+1:2*l);
pstruct.mua_0     = pvec(2*l+1:3*l);
pstruct.saa_0     = pvec(3*l+1:4*l);
pstruct.kau       = pvec(4*l+1);
pstruct.kax       = pvec(4*l+2:5*l);
pstruct.kaa       = pvec(5*l+1:6*l-1);
pstruct.omu       = pvec(6*l);
pstruct.omx       = pvec(6*l+1:7*l);
pstruct.oma       = pvec(7*l+1:8*l);

return;

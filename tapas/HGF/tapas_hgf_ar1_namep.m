function pstruct = tapas_hgf_ar1_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pstruct = struct;

l = length(pvec)/6;
    
if l ~= floor(l)
    error('Cannot determine number of levels');
end

pstruct.mu_0      = pvec(1:l);
pstruct.sa_0      = pvec(l+1:2*l);
pstruct.phi       = pvec(2*l+1:3*l);
pstruct.m         = pvec(3*l+1:4*l);
pstruct.ka        = pvec(4*l+1:5*l-1);
pstruct.om        = pvec(5*l:6*l-2);
pstruct.th        = pvec(6*l-1);
pstruct.al        = pvec(6*l);

return;

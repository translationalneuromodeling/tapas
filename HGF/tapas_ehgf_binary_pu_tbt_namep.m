function pstruct = tapas_ehgf_binary_pu_tbt_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Anna Yurova, Irene Sophia Plank, LMU University Hospital; 
% ehgf adaptation based on the function tapas_hgf_binary_pu_tbt_namep by Rebecca Lawson, Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pstruct = struct;

l = (length(pvec)-1)/5;
    
if l ~= floor(l)
    error('tapas:hgf:UndetNumLevels', 'Cannot determine number of levels');
end

pstruct.mu_0      = pvec(1:l);
pstruct.sa_0      = pvec(l+1:2*l);
pstruct.rho       = pvec(2*l+1:3*l);
pstruct.ka        = pvec(3*l+1:4*l-1);
pstruct.om        = pvec(4*l:5*l-1);
pstruct.eta0      = pvec(5*l);
pstruct.eta1      = pvec(5*l+1);

return;

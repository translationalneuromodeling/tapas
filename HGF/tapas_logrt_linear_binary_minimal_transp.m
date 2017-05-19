function [pvec, pstruct] = tapas_logrt_linear_binary_minimal_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1)     = ptrans(1);         % be0
pstruct.be0 = pvec(1);
pvec(2)     = ptrans(2);         % be1
pstruct.be1 = pvec(2);
pvec(3)     = ptrans(3);         % be2
pstruct.be2 = pvec(3);
pvec(4)     = exp(ptrans(4));    % ze
pstruct.ze  = pvec(4);

return;

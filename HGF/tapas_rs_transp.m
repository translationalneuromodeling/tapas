function [pvec, pstruct] = tapas_rs_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1)      = exp(ptrans(1));         % ze1v
pstruct.ze1v = pvec(1);
pvec(2)      = exp(ptrans(2));         % ze1i
pstruct.ze1i = pvec(2);
pvec(3)      = exp(ptrans(3));         % ze2
pstruct.ze2  = pvec(3);
pvec(4)      = exp(ptrans(4));         % ze3
pstruct.ze3  = pvec(4);

return;
function [pvec, pstruct] = tapas_sutton_k1_binary_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1)        = exp(ptrans(1));   % mu
pstruct.mu     = pvec(1);
pvec(2)        = exp(ptrans(2));   % Rhat
pstruct.Rhat   = pvec(2);
pvec(3)        = tapas_sgm(ptrans(3),1); % vhat_1
pstruct.vhat_1 = pvec(3);
pvec(4)        = exp(ptrans(4));        % h_1
pstruct.h_1    = pvec(4);

return;
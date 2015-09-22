function y = tapas_logit(x, a)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if any(x >= a) || any(x <= 0)
    error('tapas:hgf:logit:ArgOutOfRange', 'Argument out of range.');
end
    
y = log(x./(a-x));

return;
function Corr = tapas_Cov2Corr(Cov)
% Converts a covariance matrix into a correlation matrix
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Check if Cov is symmetric
if any(any(Cov'~=Cov))
    error('tapas:hgf:Cov2Corr:MatNotSymm', 'Input matrix is not symmetric.');
end

% Check if Cov is positive-definite
if any(isinf(Cov(:))) || any(isnan(Cov(:))) || any(eig(Cov)<=0)
    error('tapas:hgf:Cov2Corr:MatNotPosDef', 'Input matrix is not positive-definite.');
end

sdev = sqrt(diag(Cov));
Norm = sdev * sdev';
Corr = Cov./Norm;

return;

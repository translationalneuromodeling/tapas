%% [ ld ] = tapas_util_logdet( A )
%
% Numerical stable calculation of log-determinant for positive-definite
% matrix.
%
% INPUT:
%       A - positive definite matrix.
%
% OUTPUT:
%       ld - log(det(A))
%

% Author: Sudhir Shankar Raman
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% https://github.com/translationalneuromodeling/tapas/issues
%
function [ld] = tapas_util_logdet(A)
    
U = chol(A);
ld = 2*sum(log(diag(U)));

return;
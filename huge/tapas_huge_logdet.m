function [ld] = tapas_huge_logdet(A)
% Numerical stable calculation of log-determinant for positive-definite
% matrix.
%
% INPUT:
%   A - Positive definite matrix.
%
% OUTPUT:
%   ld - log(det(A))
%
% EXAMPLE: 
%   ld = TAPAS_HUGE_LOGDET(eye(3))    Calculate log-determinant of 3x3
%       identity matrix.
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 

U = chol(A);
ld = 2*sum(log(diag(U)));

return;
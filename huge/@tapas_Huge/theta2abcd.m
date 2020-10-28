function [A, B, C, D, tau, kappa, epsilon] = theta2abcd(theta, idx, R, L )
% Transform DCM parameters from vectorized to structured format
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
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



tmp = zeros(idx.P_f, 1);
tmp([idx.clustering; idx.homogenous]) = theta;
% hemodynamic parameters
iEd = numel(tmp);
epsilon = tmp(iEd); % ratio of intra- and extravascular signal
iEd = iEd - 1;
kappa = tmp(iEd-R+1:iEd); % decay of vasodilatory signal
iEd = iEd - R;
tau = tmp(iEd-R+1:iEd); % transit time
iEd = iEd - R;

% connectivity parameters
iSt = 0;
A = reshape(tmp(iSt+1:iSt+R^2), R, R);
iSt = iSt + R^2;
B = reshape(tmp(iSt+1:iSt+R^2*L), R, R, L);
iSt = iSt + R^2*L;
C = reshape(tmp(iSt+1:iSt+R*L), R, L);
iSt = iSt + R*L;
% nonlinear connections
if iEd - iSt == R^3
    D = reshape(tmp(iSt+1:iEd), R, R, R);
else
    D = zeros(R, R, 0);
end

end


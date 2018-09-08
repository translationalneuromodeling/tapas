function [k1, k2, k3] = tapas_mpdcm_fmri_k(theta)
%% Computes the values of k
%
% Input
% theta     -- Structure. Model parameters in mpdcm format.
%
% Output
% k1        -- Scalar.
% k2        -- Scalar.
% k3        -- Scalar.
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

r0      = 25;
nu0     = 40.3;
TE      = theta.TE;
E0      = 0.4;

k1      = 4.3*nu0*E0*TE;
k2      = exp(theta.epsilon)*r0*E0*TE;
k3      = 1 - exp(theta.epsilon);

end

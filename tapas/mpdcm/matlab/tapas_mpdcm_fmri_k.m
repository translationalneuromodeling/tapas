function [k1, k2, k3] = tapas_mpdcm_fmri_k(theta)
%% Computes the values of k 
%
% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
%
% Revision log:
%
%

r0      = 25;
nu0     = 40.3;
TE      = 0.04;
E0      = 0.4;

k1      = 4.3*nu0*E0*TE;
k2      = exp(theta.epsilon)*r0*E0*TE;
k3      = 1 - exp(theta.epsilon);

end

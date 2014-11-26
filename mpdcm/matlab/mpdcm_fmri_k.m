function [k1, k2, k3] = mpdcm_fmri_k(theta)
%% Computes the values of k 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

r0      = 25;
nu0     = 40.3;
TE      = 0.04;


k1      = 4.3*nu0*0.4*TE;
k2      = theta.epsilon*r0*0.4*TE;
k3      = 1 - theta.epsilon;


end

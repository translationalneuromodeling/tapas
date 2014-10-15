function [htheta] = mpdcm_fmri_htheta(ptheta)
%% Produces the kernel used for the proposal distribution. 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

hA = 0.001;
hB = 0.001;
hC = 0.001;

htheta = struct('c_c', []);

c_A = hA * eye(sum(true(ptheta.A)));
c_B = hB * eye(sum(true(ptheta.B)));
c_C = hC * eye(sum(true(ptheta.C)));

htheta.c_c = sparse(blkdiag(chol(c_A), chol(c_B), chol(c_C)));


end

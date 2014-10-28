function [htheta] = mpdcm_fmri_htheta(ptheta)
%% Produces the kernel used for the proposal distribution. 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

hA = 0.00001;
hB = 0.00001;
hC = 0.00001;

ht = 0.00001;
hd = 0.00001;
he = 0.00001;

hlambda = 0.00001;

htheta = struct('c_c', []);

% Evaluate the identity operator

c_A = hA * eye(sum(logical(ptheta.a(:))));
c_B = hB * eye(sum(logical(ptheta.b(:))));
c_C = hC * eye(sum(logical(ptheta.c(:))));

nr = size(ptheta.a, 1);
nu = size(ptheta.c, 2);

c_transit = ht * eye(nr);
c_decay = hd * eye(nr);
c_epsilon = he * eye(1);

c_lambda = hlambda * eye(numel(ptheta.Q));

htheta.c_c = sparse(blkdiag(chol(c_A), chol(c_B), chol(c_C), ...
    chol(c_transit), chol(c_decay), chol(c_epsilon), chol(c_lambda)));


end

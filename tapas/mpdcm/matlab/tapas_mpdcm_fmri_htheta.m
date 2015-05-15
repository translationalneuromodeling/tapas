function [htheta] = tapas_mpdcm_fmri_htheta(ptheta)
%% Produces the kernel used for the proposal distribution. 
%
% Input:
% ptheta        -- Structure. Priors of the model in mpdcm format.
%
% Output:
% htheta        -- Structure. Proposal kernel for MCMC.
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

function [ptheta] = tapas_sem_hier_prepare_ptheta(ptheta, theta, pars)
%% Thin layer of preparations for the hierarchical model. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of parameter data sets.
npars = ptheta.npars;

% Simplify the preparations
ptheta.njm = tapas_zeromat(ptheta.jm);

% Duplicate the parameters
ptheta.mu = kron(ones(npars, 1), ptheta.mu);
ptheta.pm = kron(ones(npars, 1), ptheta.pm);
ptheta.p0 = kron(ones(npars, 1), ptheta.p0);


end


function [ptheta] = tapas_sem_prosa_gaussian_priors()
%% Generates standard priors for a gaussian distribution.
%
%   Input
%   ptheta      -- Structure with the priors.
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

DIM_THETA = tapas_sem_prosa_ndims();

ptheta = struct();

[mmu, mvu, vmu, vvu, me, ve, ml, vl, p0m, p0v] = ...
    tapas_sem_unified_gaussian_priors();

mu = repmat([mmu, vmu], 1, 3);
ptheta.mu = [mu, me, ml, p0m]';

pm = repmat([mvu, vvu], 1, 3);
ptheta.pm = [1./[pm, ve, vl] p0v]';

ptheta.p0 = ptheta.mu;
ptheta.p0(9) = tan(pi * (-0.4));
% Eta is beta distributed
ptheta.bdist = [9];

end

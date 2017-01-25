function [ptheta] = tapas_sem_prosa_later_gaussian_priors()
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

mmu = 0.55;
mvu = 0.3 * 0.3;

mu = repmat([mmu, vmu], 1, 3);
ptheta.mu = [mu, me, ml, p0m]';

pm = repmat([mvu, vvu], 1, 3);
ptheta.pm = 1./[pm, ve, vl, p0v]';

ptheta.p0 = ptheta.mu;
% Eta is beta distributed
ptheta.bdist = [9];
ptheta.alpha_eta = p0v;

% Don't look at eta like a normal distributed variable
ptheta.uniform_parameters = [9];

end


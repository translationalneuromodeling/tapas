function [ptheta] = tapas_sem_seri_gaussian_priors()
%% Generates standard priors for a gaussian distribution.
%
%   Input
%   ptheta      -- Structure with the priors.
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

ptheta = struct();

[mmu, mvu, vmu, vvu, me, ve, ml, vl, p0m, p0v] = ...
    tapas_sem_unified_gaussian_priors();

mmu = 0.55;
vmu = 0.3 * 0.3;

mu = [repmat([mmu, vmu], 1, 3) 0 0];
ptheta.mu = [mu, me, ml, p0m]';

pm = [repmat([mvu, vvu], 1, 3), 1, 1];
ptheta.pm = 1./[pm, ve, vl, p0v]';

ptheta.p0 = ptheta.mu;
% Eta is beta distributed
ptheta.bdist = [11];
ptheta.alpha_eta = p0v;

% Don't look at eta like a normal distributed variable
ptheta.uniform_parameters = [7, 8, 11];

end


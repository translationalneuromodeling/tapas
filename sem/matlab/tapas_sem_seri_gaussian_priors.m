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

mu = [repmat([mmu, vmu], 1, 3) 0 0];
ptheta.mu = [mu, mu, me, ml, ml p0m]';

pm = [repmat([mvu, vvu], 1, 3), 1, 1];
ptheta.pm = 1./[pm, pm, ve, vl, vl p0v]';

ptheta.p0 = ptheta.mu;
ptheta.alpha_eta = p0v;

ptheta.uniform_parameters = [7, 8, 15, 16, 20];

end


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

mu = repmat([mmu, vmu], 1, 3);
ptheta.mu = [mu, 0.5, 0.5, me, ml, p0m]';

pm = repmat([mvu, vvu], 1, 3);
ptheta.pm = [[1./pm']; 0.5; 0.5; 1/ve ; 1/vl ; p0v];

ptheta.p0 = ptheta.mu;

% Eta is beta distributed
ptheta.bdist = [7, 8, 11];
ptheta.p0(ptheta.bdist) = tan(pi * ([0.005, 1 - 0.005, 0.005] - 0.5));

end


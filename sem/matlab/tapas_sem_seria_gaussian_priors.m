function [ptheta] = tapas_sem_seria_gaussian_priors()
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

mu = repmat([mmu, vmu], 1, 4);
ptheta.mu = [mu, me, ml, p0m]';

pm = repmat([mvu, vvu], 1, 4);
ptheta.pm = [[1./pm']; 1/ve ; 1/vl ; p0v];

ptheta.p0 = ptheta.mu;

% Eta is beta distributed
ptheta.bdist = [11];
ptheta.p0(ptheta.bdist) = tan(pi * (0.005 - 0.5));

end


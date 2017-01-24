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

ptheta.mu = [repmat([mmu, vmu], 1, 6) me, ml, ml, p0m]';
ptheta.pm = 1./[repmat([mvu, vvu], 1, 6) ve, vl, vl, p0v]';
ptheta.jm = eye(DIM_THETA); 

ptheta.alpha_eta = p0v;

ptheta.p0 = ptheta.mu;

ptheta.uniform_parameters = [16];


end


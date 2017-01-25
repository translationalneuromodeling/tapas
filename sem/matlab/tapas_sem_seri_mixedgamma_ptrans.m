function [ntheta] = tapas_sem_seri_mixedgamma_ptrans(theta)
%% Transforms the parameters to their native space 
%
% Input
%   theta       Matrix with parameters 
%
% Output
%   ntheta      Transformation of the parameters

% aponteeduardo@gmail.com
% copyright (C) 2015
%

dtheta = tapas_sem_seri_ndims();
nt = numel(theta)/dtheta;

etheta = exp(theta);
ntheta = etheta;

% Units
% invgamma
it = kron(0:nt-1, dtheta * ones(1, 2)) + kron(ones(1, nt), [1, 5]);

ntheta(it) = tapas_trans_mv2gk(etheta(it), etheta(it + 1)) + 2;
ntheta(it + 1) = tapas_trans_mv2gt(etheta(it), etheta(it + 1));

%gamma
it = kron(0:nt-1, dtheta * ones(1, 1)) + kron(ones(1, nt), [3]);
ntheta(it) = tapas_trans_mv2igk(etheta(it), etheta(it + 1));
ntheta(it + 1) = tapas_trans_mv2igt(etheta(it), etheta(it + 1));

%other units
it = kron(0:nt-1, dtheta * ones(1, 3)) + kron(ones(1, nt), [7, 8, 11]);
ntheta(it) = atan(theta(it))./pi + 0.5;

end % tapas_sem_seri_mixedgamma_ptrans 


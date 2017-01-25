function [ntheta] = tapas_sem_seri_lognorm_ptrans(theta)
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

ntheta = exp(theta);

% Units
it = kron(0:nt-1, dtheta * ones(1, 3)) + kron(ones(1, nt), [1, 3, 5]);
ntheta(it + 1) = log(ntheta(it + 1) ./ (ntheta(it).^2) + 1);
ntheta(it) = -(theta(it) - 0.5 * ntheta(it + 1));
ntheta(it + 1) = sqrt(ntheta(it + 1));

% The other parameters
it = kron(0:nt-1, dtheta * ones(1, 3)) + kron(ones(1, nt), [7, 8, 11]);
ntheta(it) = atan(theta(it))./pi + 0.5;

end % tapas_sem_seri_lognorm_ptrans 


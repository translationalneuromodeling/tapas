function [ntheta] = tapas_sem_seri_later_ptrans(theta)
%% Transform the parameters to the their native space
%
% Input
%   theta       --  Matrix with the paramers
%
% Output
%   ntheta      --  Matrix with transformed parameters
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

dtheta = tapas_sem_seri_ndims();
nt = numel(theta)/dtheta;

ntheta = theta;

% log variance to sigma parameter of a truncated normal distribution
it = kron(0:nt-1, dtheta * ones(1, 3)) + kron(ones(1, nt), [1, 3, 5]);
ntheta(it + 1) = exp(0.5 * theta(it + 1));
it = kron(0:nt-1, dtheta * ones(1, 2)) + kron(ones(1, nt), [9, 10]);
ntheta(it) = exp(theta(it));

% The other parameters
it = kron(0:nt-1, dtheta * ones(1, 3)) + kron(ones(1, nt), [7, 8, 11]);
ntheta(it) = atan(theta(it))./pi + 0.5;

end


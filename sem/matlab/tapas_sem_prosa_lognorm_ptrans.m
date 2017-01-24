function [ntheta] = tapas_sem_prosa_lognorm_ptrans(theta, dir)
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

DIM_THETA = tapas_sem_prosa_ndims();

n = 2;
if nargin < 2
    dir = 1;
end

ntheta = exp(theta);
for i = 1:(size(theta, 1)/DIM_THETA)
    offset = DIM_THETA * (i - 1);
    it = offset + [1 3 5 7 9 11];
    ntheta(it + 1) = log(ntheta(it + 1) ./ (ntheta(it).^2) + 1);
    ntheta(it) = -(theta(it) - 0.5 * ntheta(it + 1));
    ntheta(it + 1) = sqrt(ntheta(it + 1));

    it = offset + [16];
    ntheta(it) = atan(theta(it))/pi + 0.5;
end

end

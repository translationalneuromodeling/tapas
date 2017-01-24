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


DIM_THETA = tapas_sem_seri_ndims();


ntheta = exp(theta);
for i = 1:(size(theta, 1)/DIM_THETA)
    it = DIM_THETA * (i - 1) + [1 3 5 9 11 13];
    ntheta(it + 1) = log((ntheta(it + 1) ./ (ntheta(it).^2)) + 1);
    ntheta(it) = -(theta(it) - 0.5 * ntheta(it + 1));
    ntheta(it + 1) = sqrt(ntheta(it + 1));

    it = DIM_THETA * (i - 1) + [7 8 15 16 20];
    ntheta(it, :) = atan(theta(it, :))./pi + 0.5;
end


end % tapas_sem_seri_lognorm_ptrans 


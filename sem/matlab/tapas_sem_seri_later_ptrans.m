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

DIM_THETA = tapas_sem_seri_ndims();

ntheta = theta;
for i = 1:(size(theta, 1)/DIM_THETA)
    it = DIM_THETA * (i - 1) + [1 3 5 9 11 13];
    
    ntheta(it + 1, :) = exp(0.5 * theta(it + 1, :));

    it = DIM_THETA * (i - 1) + [7 8 15 16 20];
    ntheta(it, :) = atan(theta(it, :))./pi + 0.5;
end

end


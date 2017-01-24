function [ntheta] = tapas_sem_prosa_later_ptrans(theta)
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

ntheta = theta;
for i = 1:(size(theta, 1)/DIM_THETA)
    offset = DIM_THETA * (i - 1);
    it = DIM_THETA * (i - 1) + [1 3 5 7 9 11];
    
    ntheta(it + 1) = exp(0.5 * ntheta(it + 1));

    it = offset + [16];
    ntheta(it) = atan(theta(it))/pi + 0.5;
end

end % tapas_sem_prosa_gamma_ptrans 


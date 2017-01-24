function [ntheta] = tapas_sem_prosa_gamma_ptrans(theta)
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

etheta = exp(theta);
ntheta = etheta;
for i = 1:(size(theta, 1)/DIM_THETA)
    offset = DIM_THETA * (i - 1);
    it =  offset + [1 3 5 7 9 11];
    ntheta(it) = tapas_trans_mv2igk(etheta(it), etheta(it + 1));
    ntheta(it + 1) = tapas_trans_mv2igt(etheta(it), etheta(it + 1));

    it = offset + [16];
    ntheta(it) = atan(theta(it))/pi + 0.5; 
end

end % tapas_sem_prosa_gamma_ptrans 


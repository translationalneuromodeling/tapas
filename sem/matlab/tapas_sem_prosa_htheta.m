function [htheta] = tapas_sem_prosa_htheta()
%% Returns the standard hyperpriors. 
%
% Input 
%
% Output
%   htheta  -- Standard hyperpriors. htheta.pk is the precision kernel. 
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

DIM_THETA = tapas_sem_prosa_ndims();

% Precision kernel
htheta.pk = eye(DIM_THETA);
UPS = 3.0;

htheta.pk(16, 16) = UPS;

htheta.mixed = ones(DIM_THETA, 1);
htheta.mixed([16]) = 0;  


end


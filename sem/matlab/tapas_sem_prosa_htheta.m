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

dim_theta = tapas_sem_prosa_ndims();

% Precision kernel
htheta.pk = eye(dim_theta);

htheta.pk(9, 9) = 3.0;
htheta.mixed = ones(dim_theta, 1);

end

function [htheta] = tapas_sem_seri_htheta()
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

dim_theta = tapas_sem_seri_ndims();

% Precision kernel
htheta.pk = eye(dim_theta);

UPS = 3.0;

htheta.pk(7, 7) = UPS;
htheta.pk(8, 8) = UPS;

htheta.pk(11, 11) = 3.0;

htheta.mixed = ones(dim_theta, 1);
%htheta.mixed([7, 8]) = 0;

end


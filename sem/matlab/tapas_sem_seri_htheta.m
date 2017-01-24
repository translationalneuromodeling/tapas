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

DIM_THETA = tapas_sem_seri_ndims();

% Precision kernel
htheta.pk = eye(DIM_THETA);

UPS = 2.0;

htheta.pk(7, 7) = UPS;
htheta.pk(8, 8) = UPS;

htheta.pk(15, 15) = UPS;
htheta.pk(16, 16) = UPS;
htheta.pk(20, 20) = 3.0;

htheta.mixed = ones(DIM_THETA, 1);
htheta.mixed([7, 8, 15, 16, 20]) = 0;

end


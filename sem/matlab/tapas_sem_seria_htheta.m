function [htheta] = tapas_sem_seria_htheta()
%% Returns the standard hyperpriors. 
%
% Input 
%
% Output
%   htheta  -- Standard parameters of the sampler.
%               htheta.pk is the precision kernel. 


% aponteeduardo@gmail.com
% copyright (C) 2015


dim_theta = tapas_sem_seria_ndims();

% Precision kernel
htheta.pk = eye(dim_theta);

%UPS = 2.0;
htheta.pk(11, 11) = 3.0;
htheta.mixed = ones(dim_theta, 1);

end


function [llh, nx] = tapas_ti_llh_state(y, x, u, theta, ptheta)
%% Generate state space and compute the likelihood.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nx = tapas_ti_gen_state(y, x, u, theta, ptheta);
llh = tapas_ti_llh(y, nx, u, theta, ptheta);


end


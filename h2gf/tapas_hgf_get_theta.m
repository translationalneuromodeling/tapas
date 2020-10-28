function [prc, obs] = tapas_hgf_get_theta(theta, ptheta)
%% Get the parameters using indexes
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

prc = theta(ptheta.theta_prc);
obs = theta(ptheta.theta_obs);

end


function [llh] = tapas_h2gf_obs_fun(data, x, theta, ptheta)
%% A layer to compute the likelihood using the same interface used by 
% tapas_hgf
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

ptheta = ptheta.hgf;

[~, obs] = tapas_hgf_get_theta(theta, ptheta);
obs_fun = ptheta.c_obs.obs_fun;

r = ptheta;
r.y = data.y;
r.u = data.u;
r.ign = data.ign;
r.irr = data.irr;

llh = obs_fun(r, x, obs);

if any(isnan(llh))
    llh = -inf;
    return;
end

llh = sum(llh);

end


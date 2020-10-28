function [llh] = tapas_hgf_llh(y, x, u, theta, ptheta)
%% Likelihood of the hgf model. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Calculate the log-likelihood of observed responses given the perceptual 
% trajectories, under the observation model

if isnan(x)
    llh = -inf;
    return
end

try
    [~, obs] = tapas_hgf_get_theta(theta, ptheta);
    obs_fun = ptheta.obs_fun;

    llh = obs_fun(ptheta.r, x, obs);
    if any(isnan(llh))
        llh = -inf;
        return
    end
    llh = sum(llh);
    if llh == inf;
        llh = -inf;
    end
catch err
    C = strsplit(err.identifier, ':');
    if strcmp(C{1}, 'tapas')
        llh = -inf;
        return
    else
        rethrow(err)
    end  
end
end


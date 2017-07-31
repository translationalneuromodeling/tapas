function [llh] = tapas_sem_hier_llh(data, theta, ptheta)
%% Likelihood of the eye movement model in a hierarchical format. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Calculate the log-likelihood of observed responses given the perceptual 
% trajectories, under the observation model

ns = size(theta.y, 1);
nc = size(theta.y, 2);

llh = zeros(ns, nc);

for i = 1:ns
    for j = 1:nc
        tt = ptheta.model.mu + ptheta.model.jm * theta.y{i, j};
        llh(i, j) = ptheta.model.llh(data(i).y, data(i).u, {tt}, ...
            ptheta.model);
    end
end

end


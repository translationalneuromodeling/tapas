function [llh] = tapas_h2gf_llh(data, theta, ptheta)
%% Likelihood of the hgf model. 
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
         try
            % Generate the trace
            tvals = ptheta.hgf.p0 + ptheta.hgf.jm * theta.y{i, j};
            [dummy, nx] = tapas_h2gf_gen_state(data(i), tvals, ptheta);
            % Check for any error
            if any(isnan(nx))
                llh(i, j) = -inf;
            else
                llh(i, j) = tapas_h2gf_obs_fun(data(i), nx, tvals, ptheta);
            end
            if isnan(llh(i, j)) || llh(i, j) == inf
                llh(i, j) = -inf;
            end
        catch err
            C = strsplit(err.identifier, ':');
            if strcmp(C{1}, 'tapas')
                llh(i, j) = -inf;
            else
                rethrow(err)
            end  
        end   
    end
end

end


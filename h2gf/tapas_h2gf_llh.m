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
        tvals = ptheta.hgf.p0 + ptheta.hgf.jm * theta.y{i, j};
        % Try to generate a prediction of the trajectory. If
        % the error is because of the numerics of the hgf report
        % zero probability (log(0)=-infty). Other erros are thrown
        % back.
        try
            % Generate the trace
            [~, nx] = tapas_h2gf_gen_state(data(i), tvals, ptheta);
            llh(i, j) = tapas_h2gf_obs_fun(data(i), nx, tvals, ptheta);
        catch err
            c = strsplit(err.identifier, ':');
            if numel(c) < 2
                rethrow(err)
            end
            if strcmp(c{1}, 'tapas') && strcmp(c{2}, 'hgf')
                llh(i, j) = -inf;
            else
                rethrow(err)
            end
        end   
    end
end

end


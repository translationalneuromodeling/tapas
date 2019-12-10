function [posterior] = tapas_sem_hier_prepare_posterior(data, model, ...
    inference, states)
%%
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

T = model.graph{1}.htheta.T;

posterior = struct('data', data, 'model', model, 'inference', inference, ...
    'samples_theta', [], 'fe', [], 'llh', []);

np = numel(states);

theta = cell(floor(np/inference.thinning), 1);
nc = 1;
for i = 1:inference.thinning:np
    theta{nc} = states{i}.graph{2}(:, end);
    nc = nc + 1;
end

% Contains all the terms related to the likelihood
cllh = cell(2, 1);

[ns, nc] = size(states{1}.llh{1});

% Log likelihood under the posteriors
llh = zeros(ns, nc, floor(np/inference.thinning));
nc = 1;

for i = 1:inference.thinning:np
    llh(:, :, nc) = states{i}.llh{1};
    nc = nc + 1;
end

cllh{1} = llh;

posterior.llh = cllh;
if size(T, 2) > 1
    fe = trapz(T(1, :), mean(squeeze(sum(cllh{1}, 1)), 2));
else
    fe = nan;
end

posterior.fe = fe;

% Compute the WAIC / sum of the variance of the log likelihood
% (gradient of the FE) Take only the last chain (second dimension)
[waic, accuracy] = tapas_waic(squeeze(llh(:, end, :)));

posterior.waic = waic;
posterior.accuracy = accuracy;

posterior.T = T;

theta = horzcat(theta{:});
jm = model.graph{1}.htheta.model.jm;
p0 = model.graph{1}.htheta.model.p0;

for i = 1:numel(theta)
    theta{i} = p0 + jm * theta{i};
end

posterior.ps_theta = theta;

% Add the summaries automatically
posterior.summary = tapas_sem_posterior_summary(posterior);

end


function [posterior] = tapas_h2gf_prepare_posterior(data, model, ...
    inference, states)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

T = states{end}.graph{1}.T;

posterior = struct('data', data, 'model', model, 'inference', inference, ...
    'samples_theta', [], 'fe', [], 'llh', []);

np = numel(states);

theta = cell(np, 1); 
for i = 1:np
    theta{i} = states{i}.graph{2}(:, end);
end

% Contains all the terms related to the likelihood
cllh = cell(2, 1);

[ns, nc] = size(states{1}.llh{1});

% Log likelihood under the posteriors
llh = zeros(ns, nc, np);
for i = 1:np
    llh(:, :, i) = states{i}.llh{1};
end

cllh{1} = llh;

%llh = zeros(ns, nc, np);
%for i = 1:np
%    llh(:, :, i) = states{i}.llh{2};
%end
%
%cllh{2} = llh;

posterior.llh = cllh;

fe = trapz(T(1, :), mean(squeeze(sum(cllh{1}, 1)), 2));
posterior.fe = fe;

posterior.T = T;
posterior.samples_theta = theta;

% Try to stamp the version of the code

[~, hash] = tapas_get_tapas_revision(0);
posterior.tapas_revision = hash;


end


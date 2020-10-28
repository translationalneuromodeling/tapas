function [posterior] = tapas_h2gf_prepare_posterior(data, model, ...
    inference, states)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

T = model.graph{1}.htheta.T;

posterior = struct( ...
    'data', data, ...
    ... 'model', model, ... 
    'pars', inference, ... Rename to pars to better reflect the input.
    'samples', [], ... Samples of the algorithm
    'fe', [], ...
    'llh', [], ...
    'hgf', [], ... Original hgf structure
    'summary', [], ... Summary computed for the users.
    'T', [], ... Temperature schedule
    'waic',[] ... Watanabe AIC
    );

np = numel(states);

% Parameters of subjects
theta = cell(np, 1);
% Parameters of the population
theta2 = cell(np, 1);
for i = 1:np
    theta{i} = states{i}.graph{2}(:, end);
    % Mean and precision of the population
    theta2{i} = states{i}.graph{3}{1, end};
end

% Contains all the terms related to the likelihood
[ns, nc] = size(states{1}.llh{1});

% Log likelihood under the posteriors
llh = zeros(ns, nc, np);
for i = 1:np
    llh(:, :, i) = states{i}.llh{1};
end

posterior.llh = llh;
ellh = mean(squeeze(sum(llh, 1)), 2);

% Compute the model evidence using TI or WBIC
switch lower(inference.model_evidence_method)
case 'ti'
    if size(T, 2) > 1
        fe = trapz(T(1, :), ellh);
    else
        fe = nan;
    end
case 'wbic'
    % Test if the temperature makes sense
    assert(1/log(ns) == T(1, 1), ...
        'Using WBIC but the temperature is not 1/log(n)');
    fe = ellh(1);
end

% Store the free energy
posterior.fe = fe;
% Accuracy 
posterior.accuracy = ellh(end);

% Compute the WAIC / sum of the variance of the log likelihood
% (gradient of the FE)
posterior.waic = ellh(end) - sum(var(squeeze(llh(:, end, :)), [], 2));

posterior.T = T;

hgf = model.graph{1}.htheta.hgf;

posterior.samples = struct(...
    'subjects', [], ...
    'population_mean', [], ...
    'population_variance', []);

posterior.samples.subjects = horzcat(theta{:});
theta2 = [theta2{:}];
population_mean = [theta2(:).mu];
population_variance = 1./[theta2(:).pe];
for i = 1:numel(posterior.samples.subjects)
    posterior.samples.subjects{i} = tapas_h2gf_unwrapp_parameters(...
        posterior.samples.subjects{i}, hgf);
end

posterior.samples.population_mean = hgf.p0 + hgf.jm * population_mean;

% Keep the nan
p0 = hgf.p0;
p0(~isnan(p0)) = 0;
% Nan are kept as nans
posterior.samples.population_variance = p0 + hgf.jm * population_variance;

posterior.hgf = hgf;
posterior.summary = tapas_h2gf_summary(data, posterior, hgf);

end

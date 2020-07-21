function [ptheta] = tapas_h2gf_prepare_ptheta(ptheta, theta, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Check input
[nr, nc] = size(ptheta.c_prc.priormus);
assert(nc == 1 || nr == 1, 'tapas:h2gf:input', ...
    'Priors should be vectors');
if nc > 1
    ptheta.c_prc.priormus = ptheta.c_prc.priormus';
end

[nr, nc] = size(ptheta.c_prc.priorsas);
assert(nc == 1 || nr == 1, 'tapas:h2gf:input', ...
    'Priors should be vectors');
if nc > 1
    ptheta.c_prc.priorsas = ptheta.c_prc.priorsas';
end

[nr, nc] = size(ptheta.c_obs.priormus);
assert(nc == 1 || nr == 1, 'tapas:h2gf:input', ...
    'Priors should be vectors');
if nc > 1
    ptheta.c_obs.priormus = ptheta.c_obs.priormus';
end

[nr, nc] = size(ptheta.c_obs.priorsas);
assert(nc == 1 || nr == 1, 'tapas:h2gf:input', ...
    'Priors should be vectors');
if nc > 1
    ptheta.c_obs.priorsas = ptheta.c_obs.priorsas';
end

% Number of perceptual parameters
i = 1;
n = numel(ptheta.c_prc.priormus);

% Indices of perceptual parameters
ptheta.theta_prc = i : n + i -1;
% Number of perceptual parameters
ptheta.theta_prc_nd = n;

% Number of observation parameters
i = i + n;
n = numel(ptheta.c_obs.priormus);

% Indices of observation parameters
ptheta.theta_obs = i : i + n - 1;
% Number of observation parameters
ptheta.theta_obs_nd = n;

% Prior means
ptheta.mu = [ptheta.c_prc.priormus; ptheta.c_obs.priormus];
% Prior precisions
ptheta.pe = 1./[ptheta.c_prc.priorsas; ptheta.c_obs.priorsas];

% Initial sampling point
ptheta.p0 = ptheta.mu;

% Indices of sampled (ie, non-fixed) perceptual parameters
prc_idx = ptheta.c_prc.priorsas;
prc_idx(isnan(prc_idx)) = 0;
prc_idx = find(prc_idx);

% Indices of sampled (ie, non-fixed) observation parameters
obs_idx = ptheta.c_obs.priorsas;
obs_idx(isnan(obs_idx)) = 0;
obs_idx = find(obs_idx);

% Indices of all sampled (ie, non-fixed) parameters
valid = [prc_idx; numel(ptheta.c_prc.priorsas) + obs_idx];

% Projection matrix from all-parameter to sampled-parameter space
ptheta.jm = zeros(size(ptheta.mu, 1), size(valid, 1));

for i = 1:numel(valid)
    ptheta.jm(valid(i), i) = 1;
end

ptheta.jm = sparse(ptheta.jm);

% Remove from p0 the prior in all active rows. For the non active rows, 
% This line does not have an effect.
ptheta.p0 = ptheta.p0 - (ptheta.jm * ptheta.jm') * ptheta.p0;

% Stored a lower dimensional representation with only the valid or active
% parameters
ptheta.mu = ptheta.jm' * ptheta.mu;
ptheta.pe = ptheta.jm' * ptheta.pe; 

% Check for empirical prior
if ~isfield(ptheta, 'empirical_priors')
    ptheta.empirical_priors = struct();
end

% Check for weighting of the prior
if ~isfield(ptheta.empirical_priors, 'eta')
    % Defaults to 1
    ptheta.empirical_priors.eta = ptheta.jm' * ones(size(ptheta.jm, 1), 1);
elseif isscalar(ptheta.empirical_priors.eta)
    % If eta is a scalar, use the same eta for all parameters.
    ptheta.empirical_priors.eta = ptheta.jm' * ...
        ptheta.empirical_priors.eta * ...
        ones(size(ptheta.jm, 1), 1);
else
    % Otherwise it should be a vector of dimensions perceptual + observational
    ptheta.empirical_priors.eta = ptheta.jm' * ptheta.empirical_priors.eta;
end

% Hyperprior of the population mean
if ~isfield(ptheta.empirical_priors, 'alpha')
    ptheta.empirical_priors.alpha = (ptheta.empirical_priors.eta + 1)./2;
elseif isscalar(ptheta.empirical_priors.alpha)
    % If eta is a scalar, use the same eta for all parameters.
    ptheta.empirical_priors.eta = ptheta.jm' * ...
        ptheta.empirical_priors.alpha * ...
        ones(size(ptheta.jm, 1), 1);
else
    % Otherwise it should be a vector of dimensions perceptual + observational
    ptheta.empirical_priors.alpha = ptheta.jm' * ptheta.empirical_priors.alpha;
end

if ~isfield(ptheta.empirical_priors, 'beta')
    % Get the mean of the prior provided by the user.
    pe = ptheta.pe;
    % Defaults to 1
    ptheta.empirical_priors.beta = ptheta.empirical_priors.eta ./ (2.0 * pe);
elseif isscalar(ptheta.empirical_priors.beta)
    % If beta is a scalar, use the same beta for all parameters.
    ptheta.empirical_priors.beta = ptheta.jm' * ...
        ptheta.empirical_priors.beta * ...
        ones(size(ptheta.jm, 1), 1);
else
    % Otherwise it should be a vector of dimensions perceptual + observational
    ptheta.empirical_priors.beta = ptheta.jm' * ptheta.empirical_priors.beta;
end


end

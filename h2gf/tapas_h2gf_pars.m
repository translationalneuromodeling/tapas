function [pars] = tapas_h2gf_pars(data, model, pars)
%% Set up default parameters of the estimator. 
%
% Input
%       data         --
%       model       -- 
%       pars        --
% Output
%       pars        -- Updated parameters
%       

% aponteeduardo@gmail.com, chmathys@ethz.ch
% copyright (C) 2019-2020
%

% Default model evidence method is WBIC
if ~isfield('pars', 'model_evidence_method')
    pars.model_evidence_method = 'wbic';
end

% The only two model evidence methods supported are WBIC and TI
switch lower(pars.model_evidence_method)
case 'wbic'
    % Ignore supplied temperature schedule
    if isfield(pars, 'T')
        warning('tapas:h2gf:wbic', ...
            'Using WBIC: supplied temperature schedule will be ignored');
    end
    % Ignore supplied number of chains
    if isfield(pars, 'nchains')
        warning('tapas:h2gf:wbic', ...
            'Using WBIC: supplied nchains will be ignored');
    end
    % Build WBIC temperature schedule
    ns = numel(data);
    T = ones(ns, 2);
    assert(ns > 3, ...
        'WBIC is only valid when the number of subjects is more than 3');
    T(:, 1) = 1/log(ns);
    pars.T = T;
case 'ti'
    % Ignore supplied temperature schedule
    if isfield(pars, 'T')
        warning('tapas:h2gf:ti', ...
            'Using TI with power-rule temperature schedule: supplied temperature schedule will be ignored');
    end
    % Default number of chains for TI
    if ~isfield(pars, 'nchains')
        pars.nchains = 8;
    end
    % Number of chains
    nchains = pars.nchains;
    % Degree of the power schudule.
    power_rule = 5;
    % Build temperature schedule
    pars.T = (ones(size(data, 1), 1) * ...
        linspace(0.01, 1, nchains)) .^ power_rule;
otherwise
    error('tapas:h2gf:input', ...
        'model_evidence_method %s not supported', pars.model_evidence_method);
end

% Number of iterations for diagnostics
if ~isfield(pars, 'ndiag')
    pars.ndiag = 400;
end

% Default rng seed
if ~isfield(pars, 'seed')
    pars.seed = 0;
end

% Use 'shuffle' by default
if pars.seed > 0
    rng(pars.seed);
else
    rng('shuffle');
end

pars.rng_seed = rng();

% Default number of iterations kept
if ~isfield(pars, 'niter')
    pars.niter = 4000;
end

% Default number of burn-in iterations
if ~isfield(pars, 'nburnin')
    pars.nburnin = 1000;
end

% Default schedule for swapping between chains at different temperatures
% (parallel tempering)
if ~isfield(pars, 'mc3')
    pars.mc3it = 0;
end

% Default thinning (use every n-th sample)
if ~isfield(pars, 'thinning')
    pars.thinning = 1;
end

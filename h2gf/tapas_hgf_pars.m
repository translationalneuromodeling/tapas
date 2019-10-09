function [pars] = tapas_hgf_pars(data, model, pars)
%% Set up default parameters of the estimator. 
%
% Input
%       data         --
%       model       -- 
%       pars        --
% Output
%       pars        -- Updated parameters
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

if isfield(pars, 'T') && isfield(pars, 'nchains')
    warning('tapas:h2gf:input', ...
        'T and nchains are defined. nchains will be ignored.')
end

if ~isfield(pars, 'T')
    % Default of the number of chains. If the number of chains is provided
    % overwrite the default.
    nchains = 8;
    % Degree of the power schudule.
    power_rule = 5;

    if isfield(pars, 'nchains')
        nchains = pars.nchains;
    end 
    pars.T = (ones(size(data, 1), 1) * ...
        linspace(0.01, 1, nchains)) .^ power_rule;
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

% Default iterations
if ~isfield(pars, 'niter')
    pars.niter = 4000;
end

% Default burnin
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

end

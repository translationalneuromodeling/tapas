function [pars] = tapas_hgf_pars(data, model, pars)
%% 
%
% Input
%       
% Output
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

if ~isfield(pars, 'ndiag')
    pars.ndiag = 400;
end

if ~isfield(pars, 'seed')
    pars.seed = 0;
end

if pars.seed > 0
    rng(pars.seed);
else
    rng('shuffle');
end

pars.rng_seed = rng();

if ~isfield(pars, 'niter')
    pars.niter = 4000;
end

if ~isfield(pars, 'nburnin')
    pars.nburnin = 1000;
end

if ~isfield(pars, 'mc3')
    pars.mc3it = 0;
end

if ~isfield(pars, 'thinning')
    pars.thinning = 1; 
end

end

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

if ~isfield(pars, 'T')
    pars.T = ones(size(data, 1), 1) * linspace(0.01, 1, 8)' .^ 5;
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

if ~isfield(pars, 'niter')
    pars.niter = 4000;
end

if ~isfield(pars, 'nburnin')
    pars.nburnin = 1000;
end


end


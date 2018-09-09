function [pars] = tapas_linear_pars(pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isfield(pars, 'T')
    pars.T = linspace(0.01, 1, 16) .^ 5;
end

if ~isfield(pars, 'ndiag')
    pars.ndiag = 500;
end

if ~isfield(pars, 'seed')
    pars.seed = 0;
end

if pars.seed > 0
    rng(pars.seed);
else
    rng('shuffle');
end

end


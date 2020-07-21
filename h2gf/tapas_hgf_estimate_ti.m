function [ps, pars] = tapas_hgf_estimate_ti(r, prc_fun, obs_fun, pars)
%% Estiamte the hgf using the ti.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

n = 4;
if nargin < n
    pars = struct;
end

if ~isfield(pars, 'verbose')
    pars.verbose = 1;
end

if ~isfield(pars, 'mc3it')
    pars.mc3it = 0;
end

if ~isfield(pars, 'kup')
    pars.kup = 300;
end

if ~isfield(pars, 'seed')
    pars.seed = 0;
end

if ~isfield(pars, 'samples')
    pars.samples = 0;
end

if ~isfield(pars, 'T')
    pars.T = linspace(0.0001, 1, 16).^5;
end

if ~isfield(pars, 'niter')
    pars.niter = 900;
end

if ~isfield(pars, 'nburnin')
    pars.nburnin = 300;
end

if pars.seed > 0
    rng(pars.seed);
else
    rng('shuffle');
end

pars.init_htheta = @tapas_ti_init_htheta;
pars.init_ptheta = @tapas_ti_init_ptheta;
pars.init_theta = @tapas_ti_init_theta;
pars.init_y = @tapas_ti_init_y;
pars.init_u = @tapas_ti_init_u;

ptheta = tapas_hgf_ptheta();
ptheta.r = r;
ptheta.prc_fun = prc_fun;
ptheta.obs_fun = obs_fun;

htheta = tapas_hgf_htheta(ptheta);

ps = tapas_ti_estimate([], [], ptheta, htheta, pars);

end

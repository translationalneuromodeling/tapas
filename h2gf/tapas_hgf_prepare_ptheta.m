function [ptheta] = tapas_hgf_prepare_ptheta(ptheta, theta, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

n = 1;

n = n + 1;
if nargin < n
    theta  = [];
end

n = n + 1;
if nargin < n
    pars = [];
end

r = ptheta.r;

i = 1;
n = numel(r.c_prc.priormus);

ptheta.theta_prc = i : n + i -1;

i = i + n;
n = numel(r.c_obs.priormus);

ptheta.theta_obs = i : i + n - 1;

ptheta.mu = [r.c_prc.priormus; r.c_obs.priormus];
ptheta.p0 = ptheta.mu;

ptheta.jm = eye(numel(ptheta.p0));

prc_idx = r.c_prc.priorsas;
prc_idx(isnan(prc_idx)) = 0;
prc_idx = find(prc_idx);

obs_idx = r.c_obs.priorsas;
obs_idx(isnan(obs_idx)) = 0;
obs_idx = find(obs_idx);

valid = [prc_idx; numel(r.c_prc.priorsas) + obs_idx];

ptheta.jm = zeros(numel(ptheta.p0), numel(valid));
for i = 1:numel(valid)
    ptheta.jm(valid(i), i) = 1;
end

ptheta.T = pars.T;

end


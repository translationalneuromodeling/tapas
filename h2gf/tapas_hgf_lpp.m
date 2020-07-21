function [lpp] = tapas_hgf_lpp(y, ox, u, theta, ptheta)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

r = ptheta.r;
[ptrans_prc, ptrans_obs] = tapas_hgf_get_theta(theta, ptheta);


% Calculate the log-prior of the perceptual parameters.
% Only parameters that are neither NaN nor fixed (non-zero prior variance) ...
% are relevant.
prc_idx = r.c_prc.priorsas;
prc_idx(isnan(prc_idx)) = 0;
prc_idx = find(prc_idx);

logPrcPriors = -1/2.*log(2 * pi .*r.c_prc.priorsas(prc_idx)) ...
     - (0.5.*(ptrans_prc(prc_idx) - r.c_prc.priormus(prc_idx)).^2) ...
     ./r.c_prc.priorsas(prc_idx);

lpp  = sum(logPrcPriors);

% Calculate the log-prior of the observation parameters.
% Only parameters that are neither NaN nor fixed (non-zero prior variance) 
% are relevant.
obs_idx = r.c_obs.priorsas;
obs_idx(isnan(obs_idx)) = 0;
obs_idx = find(obs_idx);

logObsPriors = -1/2.*log(8 * atan(1) .* r.c_obs.priorsas(obs_idx)) ...
     - (1/2.*(ptrans_obs(obs_idx) - r.c_obs.priormus(obs_idx)).^2 ./ ...
        r.c_obs.priorsas(obs_idx));

lpp  = lpp + sum(logObsPriors);

end


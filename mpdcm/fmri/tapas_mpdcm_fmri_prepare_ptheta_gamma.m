function [ptheta] = tapas_mpdcm_fmri_prepare_ptheta_gamma(ptheta)
%% Prepares ptheta for estimation. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nb = size(ptheta.X0, 2);
nr = size(ptheta.a, 1);

s = numel(ptheta.p.theta.mu) - (nb * nr + nr - 1);
e = numel(ptheta.p.theta.mu) - (nb * nr);
ptheta.mhp(s:e) = 0;

mhp = ptheta.mhp;

pe = ptheta.p.theta.pi(mhp, mhp);
mu = ptheta.p.theta.mu(mhp);

chol_pi = chol(pe);
c = -0.5 * numel(mu) * log(2*pi) + sum(log(diag(chol_pi)));

if ptheta.X0
    ptheta.X0X0 = ptheta.X0' * ptheta.X0;
    ptheta.omega = ptheta.X0X0\ptheta.X0';
end

smu = exp(ptheta.p.theta.mu(~mhp));
spe = diag(ptheta.p.theta.pi);
spe = spe(~mhp);

ptheta.p.theta.lambda_a = smu .* spe;
ptheta.p.theta.lambda_b = spe;

% Constant of the priors
ptheta.p.theta.chol_pi = chol_pi;
ptheta.p.theta.c = c;

end


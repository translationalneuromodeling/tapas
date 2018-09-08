function [ptheta] = tapas_mpdcm_fmri_prepare_ptheta(ptheta)
%% Prepares ptheta for estimation. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

chol_pi = chol(ptheta.p.theta.pi);
c = -0.5 * numel(ptheta.p.theta.mu) * log(2*pi) + sum(log(diag(chol_pi)));

if ptheta.X0
    ptheta.X0X0 = ptheta.X0' * ptheta.X0;
    ptheta.omega = ptheta.X0X0\ptheta.X0';
    ptheta.i_chol_X0X0 = chol(inv(ptheta.X0X0))';
end

% Constant of the priors
ptheta.p.theta.chol_pi = chol_pi;
ptheta.p.theta.c = c;


end


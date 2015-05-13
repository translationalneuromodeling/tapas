function [lpp] = tapas_mpdcm_fmri_lpp(y, u, theta, ptheta)
%% Computes the log prior probability of the parameters.
%
% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
%
% Revision log:
%
%

% Assumes Gaussian priors on all the parameters.

lpp = zeros(size(theta));

nt = numel(theta);

p = tapas_mpdcm_fmri_get_parameters(theta, ptheta);

if ~isfield(ptheta.p.theta, 'chol_pi')
    chol_pi = chol(ptheta.p.theta.pi);
    c = -0.5*numel(ptheta.p.theta.mu)*log(2*pi) + sum(log(diag(chol_pi)));
else
    c = -0.5*numel(ptheta.p.theta.mu)*log(2*pi) ...
        + sum(log(diag(ptheta.p.theta.chol_pi)));
end

for i = 1:nt
    pt = p{i};
    et = pt - ptheta.p.theta.mu;
    lpp(i) =  c + -0.5 * et' * ptheta.p.theta.pi * et;
end

end

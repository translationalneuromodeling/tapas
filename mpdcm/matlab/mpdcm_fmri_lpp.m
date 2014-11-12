function [lpp] = mpdcm_fmri_lpp(y, u, theta, ptheta)
%% Computes the log prior probability of the parameters.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

% Assumes Gaussian priors on all the parameters.

lpp = zeros(size(theta));

nt = numel(theta);

p = mpdcm_fmri_get_parameters(theta, ptheta);

c = -0.5*numel(ptheta.mtheta)*log(2*pi) + sum(log(diag(ptheta.chol_ictheta)));

for i = 1:nt
    pt = p{i};
    et = pt - ptheta.mtheta;
    lpp(i) =  c + -0.5 * et' * ptheta.ictheta * et;
end

end

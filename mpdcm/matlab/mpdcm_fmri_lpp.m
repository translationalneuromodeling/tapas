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

for i = 1:nt
    pt = p{i};
    et = pt - ptheta.mtheta;
    lpp(i) = et' * ptheta.ictheta * et;
end

end

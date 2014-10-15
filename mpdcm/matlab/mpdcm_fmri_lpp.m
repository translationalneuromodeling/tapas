function [lpp] = mpdcm_fmri_lpp(y, u, theta, ptheta)
%% Computes the log prior probability of the parameters.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

% Assumes Gaussian priors on all the parameters.

lpp = zeros(size(theta));

nt = numel(theta);

p = mpdcm_fmri_get_parameters(theta);

for i = 1:nt
    ttheta = p{i};
    e = pt - ptheta.mtheta;
    lpp(i) = (tp' * ptheta.ctheta * tp;
end

end

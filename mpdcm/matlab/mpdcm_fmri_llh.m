function [llh] = mpdcm_fmri_llh(y, u, theta, ptheta)
%% Computes the likelihood of the data.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

% Integrates the system

ny = mpdcm_fmri_llh(u, theta, ptheta);

% Compute the likelihood

nt = numel(theta);
ny = numel(ny);

llh = zeros(size(theta));

for i = 1:numel(theta)
    theta0 = theta{i};
    y0 = y{mod(i, ny)};
    ny0 = y{i);

    e = y0 - ny0;

    llh(i) = e' * theta.yC * e;
end


end

function [llh] = mpdcm_fmri_llh(y, u, theta, ptheta, sloppy)
%% Computes the likelihood of the data.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 5
    sloppy = 0;
end

% Integrates the system

ny = mpdcm_fmri_int(u, theta, ptheta, sloppy);

% Computes the likelihood

s1 = size(theta, 1);
s2 = size(theta, 2);

llh = zeros(size(theta));

for i = 1:s1
    y0 = y{s1}(:);
    for j = 1:s2
        theta0 = theta{(i-1)*s1 + j};
        if eigs(theta0.A, 1) > 0
            llh((i-1)*s1+j) = -inf;
        end
        ny0 = ny{(i-1)*s1 + j}(:);

        e = y0 - ny0;

        % Compute the hyperpriors

        tlambda = exp(theta0.lambda);

        nQ = tlambda(1) * ptheta.Q{1};
        for k = 2:numel(ptheta.Q)
            nQ = nQ + tlambda(k) * ptheta.Q{k};
        end

        llh((i-1)*s1+j) = -0.5 * e' * (nQ\e);
    end
end

end

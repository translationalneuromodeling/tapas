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
        ny0 = ny{(i-1)*s1 + j}(:);

        e = y0 - ny0;

        % Compute the hyperpriors

        tlambda = ones(1, 1, size(ptheta.Q, 3));
        tlambda(:) = exp(theta0.lambda);

        %tQ = sum(bsxfun(@times, ptheta.Q, tlambda), 3);

        llh((i-1)*s1+j) = e' * e;
    end
end

end

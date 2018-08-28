function [llh] = tapas_dlinear_llh(data, theta, ptheta)
%% Likelihood function of a linear model where we assume that the covariance
% is diagonal. Note that this is the prior of a Gamma Gaussian model.

%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

y = data.y;
u = data.u;

theta = theta.y;

[~, nc] = size(y);

llh = zeros(1, nc);
ln2pi = log(2 * pi);

for i = 1:nc
    pe = y{i}.pe;
    lpe = log(pe);
    % Likelihood of the precision
    tllh = theta{i}.alpha .* log(theta{i}.beta) - ...
        gammaln(theta{i}.alpha) - ...
        pe .* theta{i}.beta + ...
        (theta{i}.alpha - 1) .* lpe;

    r = theta{i}.mu - y{i}.mu;
    llh(1, i) = sum(tllh - 0.5 * ln2pi + 0.5 * lpe ...
        - 0.5 * pe .* r .* r);
end

end

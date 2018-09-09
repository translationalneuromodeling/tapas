function [llh] = tapas_mdlinear_llh(data, theta, ptheta)
%% Likelihood function of a multivariate linear model where we assume that
% the covariances are diagonal. 
%
% This is the likelihood of a Gamma Gauss model % of the form
%
% p(beta| beta0, pe)p(pe) = N(beta; X*beta0, 1/pe) Gamma(pe; a, b)
%
% Note thet beta is a matrix were beta each vector correspond to the number
% of parameters fitted and the rows to the number of regressors such that
% 
%   x * beta
%
% gives a matrix of n observations and m parameters. pe is a vector of 
% precisions of size m. 
%

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
        (theta{i}.alpha - 1) .* log(pe);

    r = theta{i}.mu - y{i}.mu;
    [nr, np] = size(r); 
    llh(1, i) = sum(tllh - 0.5 * nr * ln2pi + 0.5 * nr * lpe ...
        - 0.5 * pe .* sum(r .* r, 1));
end

end

function [waic, lppd] = tapas_waic(llh)
%% Computes the Watanabe-Akaike Information criterion
%
% Input
%       llh         -- An NxS array containing the samples of the
%                       likelihood where N is the number of subjects and S
%                       the number of samples.
%
% Output
%       waic        -- The Watanabe AIC usin the methods in [1].
%       lppd    -- The log pointwise predictive density of the model.
%
% The WAIC can be computed as the lppd - penalization, where the
% penalization is the sum of the variance of the log likelihood, i.e., the
% gradient of the Free energy.
%
% [1] Understanding predictive Information Criteria for Bayesian Models 
%   Gelman et al.
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%
% REVISION LOG:
%
%      Jakob Heinzle, 2021/04/16: changed computation of accuracy to be exactly log
%      pointwise predictive density as in Gelman et al.
%
%%

s = size(llh,2);

% Estimator of the accuracy
lppd = sum(tapas_logsumexp(llh')-log(s));

% Estimator of the variance
penalization = sum(var(llh, [], 2));

waic = lppd - penalization;

end

function [waic, accuracy] = tapas_waic(llh)
%% Computes the Watanabe-Akaike Information criterion
%
% Input
%       llh         -- An NxS array containing the samples of the
%                       likelihood where N is the number of subjects and S
%                       the number of samples.
%
% Output
%       waic        -- The Watanaba AIC usin the methods in [1].
%       accuracy    -- The expected log likelihood of the model.
%
% The WAIC can be computed as the accuracy - penalization, where the
% penalization is the sum of the variance of the log likelihood, i.e., the
% gradient of the Free energy.
%
% [1] Understanding predictive Information Criteria for Bayesian Models 
%   Gelman et al.
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

% Estimator of the accuracy
accuracy = sum(mean(llh, 2));

% Estimator of the variance
penalization = sum(var(llh, [], 2));

waic = accuracy - penalization;

end

function [gt] = tapas_trans_mv2gt(mu, sigma2)
%% Transforms to the scaling of the gamma distribution
%
% Input
%   mu          -- Means
%   sigma2      -- Variance
%
% Output
%   gk          -- K parameter of an gamma distribution

% aponteeduardo@gmail.com
% copyright (C) 2015
%

gt = mu ./ sigma2;

end 

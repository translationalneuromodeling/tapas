function [gk] = tapas_trans_mv2gk(mu, sigma2)
%% Transforms to the scaling of the gamma distribution%
% Input
%   mu          -- Means
%   sigma2      -- Variance
%
% Output
%   gk          -- K parameter of an inverse gamma distribution

% aponteeduardo@gmail.com
% copyright (C) 2015
%

gk = mu.^2 ./ sigma2;

end 
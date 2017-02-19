function [gk] = tapas_trans_mv2igk(mu, sigma2)
%% Transforms the parameters to their native space 
%
% Input
%   mu          -- Means
%   sigma2      -- Variance
%
% Output
%   gk          -- K parameter of an inverse gamma distribution

% aponteeduardo@gmail.com
% copyright (C) 2015
%

gk = (mu.^2 ./ sigma2) + 2;

end 
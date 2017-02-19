function [gt] = tapas_trans_mv2igt(mu, sigma)
%% Transforms to the scaling of the inverse gamma distribution
%
% Input
%   theta       Matrix with parameters 
%
% Output
%   ntheta      Transformation of the parameters

% aponteeduardo@gmail.com
% copyright (C) 2015
%

gt = 1./((mu .^ 3  ./ sigma ) + mu);

end 

function [alpha, beta] = tapas_gamma_moments_to_ab(m1, m2)
%% Transforms the central moments of a Gamma distribution into the alpha beta 
% parametrization where x^alpha-1 b^alpha exp(-x * beta)
%
% Input
%   m1      -- First central moment (mean)
%   m2      -- Second central momement (variance)
%       
% Output
%   alpha   -- Alpha parameter
%   beta    -- Beta parameters (scale)
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

beta = m1 ./ m2;
alpha = beta .* m1;


end


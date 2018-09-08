function [ntheta] = tapas_mpdcm_erp_priors(theta)
%% Given a model returns the standard priors
%
% Input
%   theta           spm structure M.pE.E
%
% Output
%   ntheta          spm new structure M.pE.E

% aponteeduardo@gmail.com
% copyright (C) 2016
%

ntheta = theta;
ntheta.A{1} = log(32) + 0 * theta.A{1};
ntheta.A{2} = log(16) + 0 * theta.A{2};
ntheta.A{3} = log(4) + 0 * theta.A{3};

g1 = 128;

ntheta.H = log([g1 (g1*4/5) (g1/4) (g1/4)]);
ntheta.G(:, 1) = log(4) + 0 * theta.G(:, 1);
ntheta.G(:, 2) = log(32) + 0 * theta.G(:, 2);

ntheta.T(:, 1) = log(8) + 0 * theta.T(:, 1);
ntheta.T(:, 2) = log(16) + 0 * theta.T(:, 2);

ntheta.S = [log(2/3) log(1/3)] + 0 * theta.S;


end % tapas_mpdcm_erp_priors 


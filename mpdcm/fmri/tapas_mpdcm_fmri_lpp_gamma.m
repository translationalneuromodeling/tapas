function [lpp] = tapas_mpdcm_fmri_lpp_gamma(y, u, theta, ptheta)
%% Computes the log prior probability of the parameters.
%
% Input:
% y         -- Cell array. Each cell contains the experimental data.
% u         -- Cell array. Each cell contains the model input.
% theta     -- Cell array. Each cell contains the model parameters.
% ptheta    -- Structure. Prior Qf all models.  % sloppy    -- Scalar. If 0 input is not checked. Defaults to 1.  % % Output:
% lpp       -- Cell array. Each cell contains an scalar with the log prior 
%           probability.
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

% Assumes Gaussian priors on all the parameters.

lpp = zeros(size(theta));

nt = numel(theta);

p = tapas_mpdcm_fmri_get_parameters(theta, ptheta);

c = ptheta.p.theta.c;

mhp = ptheta.mhp;
mu = ptheta.p.theta.mu(mhp);
pe = ptheta.p.theta.pi(mhp, mhp);

for i = 1:nt
    pt = p{i}(mhp);
    et = pt - mu;
    lpp(i) =  c + -0.5 * et' * pe * et;
    
    pt = p{i}(~mhp);
    pt = -ptheta.p.theta.lambda_b .* exp(pt) + ...
        (ptheta.p.theta.lambda_a - 1) .* pt;

   lpp(i) = lpp(i) + sum(pt);

end

end

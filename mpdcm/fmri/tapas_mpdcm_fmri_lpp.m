function [lpp] = tapas_mpdcm_fmri_lpp(y, u, theta, ptheta, p)
%% Computes the log prior probability of the parameters.
%
% Input:
% y         -- Cell array. Each cell contains the experimental data.
% u         -- Cell array. Each cell contains the model input.
% theta     -- Cell array. Each cell contains the model parameters.
% ptheta    -- Structure. Prior of all models.
% sloppy    -- Scalar. If 0 input is not checked. Defaults to 1.
%
% Output:
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

if nargin < 5
    p = tapas_mpdcm_fmri_get_parameters(theta, ptheta);
end

if ~isfield(ptheta.p.theta, 'c')
    chol_pi = chol(ptheta.p.theta.pi);
    c = -0.5*numel(ptheta.p.theta.mu)*log(2*pi) + sum(log(diag(chol_pi)));
else
    c = ptheta.p.theta.c;
end

for i = 1:nt
    pt = p{i};
    et = pt - ptheta.p.theta.mu;
    lpp(i) =  c + -0.5 * et' * ptheta.p.theta.pi * et;
end

end

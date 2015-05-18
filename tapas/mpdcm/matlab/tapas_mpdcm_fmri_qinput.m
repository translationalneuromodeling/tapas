function [ny, nu, ntheta, nptheta] = tapas_mpdcm_fmri_qinput(y, u, theta, ...
ptheta)
%% Expand the input for a simplified version of the Laplace approximation
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

% Check input

assert(size(y{1}, 1) > 1, 'mpdcm:fmri:gmodel:input', ...
    'Single region models are not implemented');
tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);

ny = y;
nu = u;
ntheta = theta;
nptheta = ptheta;

nr = theta{1}.dim_x;

e_lambda = ptheta.p.theta.mu(end-nr+1:end);
c_lambda = diag(ptheta.p.theta.sigma);
c_lambda = c_lambda(end-nr+1:end);
c_lambda = c_lambda(:);

% Moment generating function of a gaussian, equivalent to the mean an second
% moment of a log normal

m1 = exp(e_lambda + 0.5*c_lambda);
m2 = exp(2*e_lambda + 0.5*c_lambda*4);

nptheta.p.lambda.b = 1./((m2./m1) - m1);
nptheta.p.lambda.a = m1.*nptheta.p.lambda.b;

for i = 1:numel(u)

    ntheta{i}.q.theta.mu = [];
    ntheta{i}.q.theta.pi_theta = [];

    % Use the moment generating function to approximate the prior with a Gamma 
    % distribution up to its two first moments.

    % Use the fact that the scale parameter is equal to variance divided by the 
    % mean and that the variance is equal to the second momment minus the first 
    % moment to the power of 2


    ntheta{i}.q.lambda.a = nptheta.p.lambda.a;
    ntheta{i}.q.lambda.b = nptheta.p.lambda.b;
end

end


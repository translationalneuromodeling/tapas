function [ny, nu, ntheta, nptheta] = mpdcm_fmri_qinput(y, u, theta, ptheta)
%% Expand the input for a simplified version of the Laplace approximation
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

ny = y;
nu = u;
ntheta = theta;
nptheta = ptheta;

ntheta.q.theta.mu = [];
ntheta.q.theta.pi_theta = [];

% Use the moment generating function to approximate the prior with a Gamma 
% distribution up to its two first moments.

nr = theta.dim_x;

e_lambda = ptheta.mtheta(end-nr+1:end);
c_lambda = diag(ptheta.ctheta);
c_lambda = c_lambda(end-nr+1:end);
c_lambda = c_lambda(:);

% Moment generating function of a gaussian, equivalent to the mean an second
% moment of a log normal

m1 = exp(e_lambda + 0.5*c_lambda);
m2 = exp(2*e_lambda + 0.5*c_lambda*4);

% Use the fact that the scale parameter is equal to variance divided by the 
% mean and that the variance is equal to the second momment minus the first 
% moment to the power of 2

nptheta.p.lambda.b = (m2./m1) - m1;
nptheta.p.lambda.a = m1./nptheta.p.lambda.b;

ntheta.q.lambda.a = nptheta.p.lambda.a;
ntheta.q.lambda.b = nptheta.p.lambda.b;

end


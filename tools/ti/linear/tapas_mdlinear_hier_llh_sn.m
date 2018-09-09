function [llh] = tapas_mdlinear_hier_llh_sn(data, theta, ptheta, s1, s2i, ...
    s2j)
%% Likelihood of the nodes of a linear multivariate model with diagonal 
% precisions on each parameter. 
%
% It is the likelihood of 
%
% p(y|x, beta, pi) = N(y - x*b, 1/pi)
%
% Note that this is a multivariate regression problem, so beta is a matrix
% in which each column correspond to one parameter of the model.
%

%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

y = data.y;
u = data.u;

x = u.x;
theta = theta.y;

[np, nc] = size(y);

ln2pi = log(2 * pi);

% Mean
beta = theta{s2i}.mu;
% Precision
pe = theta{s2i}.pe;

lpe = log(pe);

% number of subjects x number of parameters
py = (x(s1, :) * beta)';
inp = numel(py);
ty = y{s1, s2j};

% Residuals
r = (ty - py)';

% Prediction error 
e = - 0.5 * pe .* r .* r;
llh = - inp * 0.5 * ln2pi + 0.5 * sum(lpe) + sum(e);

end


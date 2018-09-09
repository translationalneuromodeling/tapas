function [llh] = tapas_dlinear_hier_llh_sn(data, theta, ptheta, s1, s2i, s2j)
%% Likelihood of the nodes of a model with diagonal precision matrix in a 
% single cell.
%
% Note that this is the likelihood of several parameters theta 1 to n with mean
% mu and precision matrix diag(pe). This is applied to the theta in the cell
% j to the data set in i, k
%

%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

y = data.y;
u = data.u;

theta = theta.y;
ln2pi = log(2 * pi);

% Mean
mu = theta{s2i}.mu;
% Precision
pe = theta{s2i}.pe;
lpe = log(pe);

r = y{s1, s2j} - mu;
llh = sum(- 0.5 * ln2pi + 0.5 * lpe - 0.5 * pe .* r .* r);

end

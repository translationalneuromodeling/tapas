function [ptheta] = tapas_sem_hier_prepare_ptheta(ptheta, theta, pars)
%% Thin layer of preparations for the hierarchical model. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nvals = size(ptheta.jm, 1);
ndims = ptheta.ndims;

npars = nvals / ndims;

assert(floor(npars) == npars, ...
    'tapas:sem:hier:ptheta', ...
    ['Dimensions of the constraint matrix is %dx%d ' ...
    'but parameters of model %s are %d'], ...
    size(ptheta.jm, 1), size(ptheta.jm, 2), ptheta.name, ndims);

ptheta.npars = npars;

% Rename
ptheta.name = sprintf('hier_%s', ptheta.name);

% Simplify the preparations
ptheta.njm = tapas_zeromat(ptheta.jm);

% Duplicate the parameters
ptheta.mu = kron(ones(npars, 1), ptheta.mu);
ptheta.pm = kron(ones(npars, 1), ptheta.pm);
ptheta.p0 = kron(ones(npars, 1), ptheta.p0);

end

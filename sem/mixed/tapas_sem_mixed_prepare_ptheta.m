function [ptheta] = tapas_sem_mixed_prepare_ptheta(ptheta, theta, pars)
%% Thin layer of preparations for the hierarchical model. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of parameter data sets.
nvals = size(ptheta.jm, 1);
ndims = ptheta.ndims;

npars = nvals / ndims;

assert(floor(npars) == npars, ...
    'tapas:sem:hier:ptheta', ...
    ['Dimensions of the constraint matrix is %dx%d ' ...
    'but parameters of model %s are %d'], ...
    size(ptheta.jm, 1), size(ptheta.jm, 2), ptheta.name, ndims);

% Rename
ptheta.name = sprintf('mixed_%s', ptheta.name);

% Simplify the preparations
ptheta.njm = tapas_zeromat(ptheta.jm);

% Duplicate the parameters
ptheta.mu = kron(ones(npars, 1), ptheta.mu);
ptheta.pm = kron(ones(npars, 1), ptheta.pm);
ptheta.p0 = kron(ones(npars, 1), ptheta.p0);

% Precompute some of the linear elements.
xx = ptheta.x' * ptheta.x;
if rank(xx) < size(xx, 1)
    warning('The design matrix is singular.');
end
ptheta.omega = xx + eye(size(ptheta.x, 2));
ptheta.iomega = inv(ptheta.omega);
ptheta.comega = chol(ptheta.omega);
ptheta.ciomega = chol(ptheta.iomega);

end


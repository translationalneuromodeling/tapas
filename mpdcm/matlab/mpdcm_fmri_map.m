function [mu, ny, dfdx] = mpdcm_fmri_map(y, u, theta, ptheta, mleflag)
%% Minimizes map estimate assuming fixed noise.
%
% Input:
%
% y -- Cell array of experimental observations.
% u -- Cell array of experimental design.
% theta -- Cell array of initialization model parameters.
% ptheta -- Structure of the hyperparameters.
% mleflag -- If true, computes only the maximum likelihood estimator. Defaults
%   to 0;
%
% Output:
%
% mu -- Maximum of mu
% ny -- Predicted signal at mu
% dfdx -- Derivatives evaluated in dfdx
%
% It uses weighted regularized Levenberg-Marquard Gaussian-Newton 
% optimization.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 5 
    mleflag = 0;
end

% Tolereance to deviation
tol = 1e-3;
dt = 1e-4;

% Multiplies the gradient with respect to the priors
vdpdx = 1;

if mleflag
    vdpdx = 0;
end

assert(size(theta, 1) == 1, 'mpdcm:fmri:mle:input', ...
    'theta second dimension should be one, instead %d', size(theta, 2));

mpdcm_fmri_int_check_input(u, theta, ptheta);

% Initilize
op = mpdcm_fmri_get_parameters(theta, ptheta);
np = op;

su = size(u);

% Ensamble the weights. Only diagonal matrices are accepted.

nQ = cell(su);

nctheta = 1./diag(ptheta.ctheta);
nctheta(end-size(ptheta.a, 1):end) = 0;

for i = 1:numel(u)
    tlambda = exp(theta{i}.lambda);
    tQ = zeros(size(ptheta.dQ.Q{1}));
    for k = 1:numel(ptheta.dQ.Q)
        tQ = tQ + tlambda(k)*ptheta.dQ.Q{k};
    end
    
    nQ{i} = [tQ(:); nctheta];
end

% Regularization parameter

lambda0 = 0.5 * ones(su);
lambda = 0.5 * ones(su);
v = 1.3 * ones(su);
reg = 1 * ones(su);

% Objective function

ors = inf * ones(su);
nrs = zeros(su);

% Error array

e = cell(su);

for j = 1:100

    if mod(j, 10) == 0
        fprintf(1, 'Iteration: %d, RSE: ', j);
        fprintf(1, '%0.5f ', nrs);
        fprintf(1, '\n');
    end
    
    [dfdx, ny] = mpdcm_fmri_gradient(op, u, theta, ptheta, 1);

    for k = 1:numel(u)
        ny = ny{k};
        e{k} = y{k}' - ny;
        e{k} = [e{k}(:); ptheta.mtheta - op{k}];
    end

    lambda = lambda0.*v.^reg;

    for k = 1:numel(dfdx)
        tdfdx = dfdx{k};
        tdfdx = reshape(tdfdx, size(tdfdx, 1) * size(tdfdx, 2), 1, ...
            size(tdfdx, 3));
        dfdx{k} = squeeze(tdfdx);
    end

    for k = 1:numel(u)
        tdfdx = cat(1, dfdx{k}, vdpdx * eye(numel(op{k})));
        np{k} = op{k} + (tdfdx'*bsxfun(@times, tdfdx, nQ{k}) + ...
            lambda(k) * eye(numel(op{k})))\(tdfdx'*(nQ{k}.*e{k}));
    end

    ntheta = mpdcm_fmri_set_parameters(np, theta, ptheta);

    ny = mpdcm_fmri_int(u, ntheta, ptheta);

    % Verify the residuals
    for k = 1:numel(u)
        e{k} = y{k}' - ny{k};
        e{k} = e{k}(:);
        nrs(k) = e{k}'*e{k};
    end

    if all(abs(nrs - ors) < tol)
        op(nrs < ors) = np(nrs < ors);
        break;
    end

    nrs(isnan(nrs)) = inf;

    reg(nrs >= ors) = reg(nrs >= ors) + 1;
    reg(nrs < ors) = reg(nrs < ors) - 1;
    op(nrs < ors) = np(nrs < ors);
    ors(nrs < ors) = nrs(nrs < ors);

    assert(any(ors < inf), 'mpdcm:fmri:mle:numeric', ... 
       'Optimization failed with no gradient being found.')

end

mu = op;
[dfdx, ny] = mpdcm_fmri_gradient(op, u, theta, ptheta, 1);

end

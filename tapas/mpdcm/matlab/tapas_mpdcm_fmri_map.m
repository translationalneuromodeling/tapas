function [mu, ny, dfdx, nrs] = tapas_mpdcm_fmri_map(y, u, theta, ptheta, pars)
%% Minimizes map estimate assuming fixed noise.
%
% Input:
%
% y         -- Cell array of experimental observations.
% u         -- Cell array of experimental design.
% theta     -- Cell array of initialization model parameters.
% ptheta    -- Struct. containing the hyperparameters.
% pars      -- Struct. mleflag: If true, computes only the maximum likelihood 
%               estimator. Defs. to 0. verb: Verbose output. Defs. 0.
%
% Output:
%
% mu -- Maximum of mu
% ny -- Predicted signal at mu
% dfdx -- Derivatives evaluated in dfdx
% nrs -- Last value of the objective function for each data set.
%
% It uses weighted regularized Levenberg-Marquard Gaussian-Newton 
% optimization.
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

assert(size(theta, 2) == 1, 'mpdcm:fmri:mle:input', ...
    'theta second dimension should be one, instead %d', size(theta, 2));

if nargin < 5
    pars = struct();
end

if ~isfield(pars, 'mleflag')  
    pars.mleflag = 0;
end

if ~isfield(pars, 'verb')  
    pars.verb = 0;
end

% Defaults to no temperature parameter.

if ~isfield(ptheta, 'T')
    ptheta.T = ones(numel(y), 1);
end

% Tolereance to deviation
tol = 1e-3;
dt = 1e-4;

% Multiplies the gradient with respect to the priors
vdpdx = 1;

if pars.mleflag
    vdpdx = 0;
end

tapas_mpdcm_fmri_int_check_input(u, theta, ptheta);

% Initilize
op = tapas_mpdcm_fmri_get_parameters(theta, ptheta);
np = op;

su = size(u);

% Ensamble the weights. Only diagonal matrices are accepted.

nQ = cell(su);

nctheta = diag(ptheta.p.theta.pi);
nctheta(end-size(ptheta.a, 1)+1:end) = 0;

for i = 1:numel(u)
    tlambda = exp(theta{i}.lambda);

    tQ = zeros(size(ptheta.dQ.Q{1}));
    for k = 1:numel(ptheta.dQ.Q)
        tQ = tQ + tlambda(k)*ptheta.dQ.Q{k};
    end 
    % Weight data by temperature
    nQ{i} = [ptheta.T(i) * tQ(:); nctheta];
end

% Regularization parameter

lambda0 = 0.5 * ones(su);
lambda = 0.5 * ones(su);
v = 1.1 * ones(su);
reg = 1 * ones(su);

% Objective function

ors = inf * ones(su);
nrs = zeros(su);

% Error array

e = cell(su);
dp = zeros(su);

for j = 1:30

    [dfdx, ny] = tapas_mpdcm_fmri_gradient(op, u, theta, ptheta, 1);

    for k = 1:numel(u)
        e{k} = y{k}' - ny{k};
        % There might be subject dependent priors
        e{k} = [e{k}(:); ptheta.p.theta.mu(:, ...
            min(k, size(ptheta.p.theta.mu, 2))) - op{k}];
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
        assert(isreal(np{k}), 'Non real values');
        assert(~all(isnan(np{k})), 'Undefined value');
    end

    ntheta = tapas_mpdcm_fmri_set_parameters(np, theta, ptheta);

    ny = tapas_mpdcm_fmri_int(u, ntheta, ptheta, 1); 

    % Verify the residuals
    for k = 1:numel(u)
        e{k} = y{k}' - ny{k};
        e{k} = e{k}(:);
        nrs(k) = e{k}'*e{k};
        dp(k) = max(abs(np{k} - op{k}));
    end

    nrs(isnan(nrs)) = inf;

    if all(dp < tol)
        op(nrs < ors) = np(nrs < ors);
        break;
    end

    reg(nrs >= ors) = reg(nrs >= ors) + 1;
    reg(nrs < ors) = reg(nrs < ors) - 1;
    op(nrs < ors) = np(nrs < ors);
    ors(nrs < ors) = nrs(nrs < ors);

    assert(any(ors < inf), 'mpdcm:fmri:mle:numeric', ... 
       'Optimization failed with no gradient being found.')

end

if pars.verb
    fprintf(1, 'MAP Iteration: %d, RSE: ', j);
    fprintf(1, '%0.1f ', nrs);
    fprintf(1, '\n');
end

mu = op;
assert(isreal(cell2mat(mu)), 'Non real values');
[dfdx, ny] = tapas_mpdcm_fmri_gradient(mu, u, theta, ptheta, 1);

end


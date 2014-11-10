function [mX, cX] = mpdcm_fmri_mle(y, u, theta, ptheta)
%% Minimizes the likelihood function of a dcm for fmri.
%
% Input:
%
% y -- Cell array of experimental observations.
% u -- Cell array of experimental design.
% theta -- Cell array of initialization model parameters.
% ptheta -- Structure of the hyperparameters.
%
% Output:
%
% mX -- Approximated output
% mC -- Estimated variance.
%
% It uses weighted regularized Levenberg-Marquard Gaussian-Newton 
% optimization.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

% Tolereance to deviation
tol = 1e-2;
dt = 1e-3;

assert(size(theta, 1) == 1, 'mpdcm:fmri:mle:input', ...
    'theta second dimension should be one, instead %d', size(theta, 2));

mpdcm_fmri_int_check_input(u, theta, ptheta);

% Initilize
op = mpdcm_fmri_get_parameters(theta, ptheta);
np = op;

su = size(u);

% Ensamble the weights. Only diagonal matrices are accepted.

nQ = cell(su);

for i = 1:numel(u)
    tlambda = exp(theta{i}.lambda);
    tQ = zeros(size(ptheta.dQ.Q{1}));
    for k = 1:numel(ptheta.dQ.Q)
        tQ = tQ + tlambda(k)*ptheta.dQ.Q{k};
    end
    nQ{i} = tQ(:);
end

% Regularization parameter

lambda0 = 0.1 * ones(su);
lambda = 0.1 * ones(su);
v = 1.1 * ones(su);
k = 1 * ones(su);

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
        e{k} = e{k}(:);
    end

    lambda = lambda0.*v.^k;

    for k = 1:numel(dfdx)
        dfdx{k} = dfdx{k}(:);
    end

    for k = 1:numel(u)
        tdfdx = cell2mat(dfdx(k, :));
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

    if all(nrs - ors) < tol
        op(nrs < ors) = np(nrs < ors);
        break;
    end

    k(nrs >= ors) = k(nrs >= ors) + 1;
    k(nrs < ors) = k(nrs < ors) - 1;
    op(nrs < ors) = np(nrs < ors);
    ors(nrs < ors) = nrs(nrs < ors);

    %if all(abs(dp) < 1e-3)
    %    break;
    %end

end

mX = op;
cX = cell(su);
for i = 1:numel(u)
    cX{i} = var(reshape(e{i}, size(ny{i})));
end
end

function [q, ntheta] = tapas_mpdcm_fmri_gmodel(y, u, theta, ptheta, pars) 
%% Estimate the model using an approximate variational Bayes method
% 
% Input:
% y         -- Cell array of experimental observations.
% u         -- Cell array of experimental input.
% theta     -- Cell array of initial states of the parameters
% ptheta    -- Priors and constants
% pars      -- Struct. Parameters for the optimization.
%
% Output:
% q         -- Sufficient statistics of the approximate marginal posteriors.
% ntheta    -- MAP estimator of the model
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

if nargin < 5
    pars = struct();
end

if ~isfield(pars, 'verb')
    pars.verb = 0;
end

if ~isfield(pars, 'maxi')
    pars.maxi = 5;
end

TOL = 1e-6;

assert(size(y{1}, 2) > 1, 'tapas:mpdcm:fmri:gmodel:input', ...
    'Single region models are not implemented');

[y, u, theta, ptheta] = tapas_mpdcm_fmri_qinput(y, u, theta, ptheta);

ns = size(y{1}, 2);
nr = size(y{1}, 1);
np = numel(ptheta.p.theta.mu);
nt = numel(theta);

q = cell(nt, 1);

for j = 1:nt
    q{j} = theta{j}.q;
end

% Optimize the posterior
for i = 1:pars.maxi
    for j = 1:nt
        theta{j}.lambda = full(log(q{j}.lambda.a) - log(q{j}.lambda.b));
    end
    [mu, ny, dfdx] = tapas_mpdcm_fmri_map(y, u, theta, ptheta, pars);

    theta = tapas_mpdcm_fmri_set_parameters(mu, theta, ptheta);

    for j = 1:nt
        q{j}.theta.mu = mu{j};
        % Approximate hessian
        % Weight the hessian by the precisions
        dfdx{j} = dfdx{j}(:,:,1:np-nr);

        h1 = bsxfun(@times, dfdx{j}, reshape(exp(theta{j}.lambda), 1, nr, 1));
        h1 = reshape(h1, ns*nr, 1, np-nr);
        h1 = squeeze(h1);

        h2 = reshape(dfdx{1}, ns*nr, 1, np-nr);
        h2 = squeeze(h2);

        q{j}.theta.pi = h1'*h2; 
        
        % Estimate the noise

        q{j}.lambda.a(:) = 0.5 * ns + ptheta.p.lambda.a - 1;

        e = ny{j} - y{j}';
        for k = 1:nr
            h = dfdx{1}(:, k, :);
            h = squeeze(h);
            h = h'*h;
            % h is probably a low dimensional rank matrix so we can transport
            % it to a new space
            [v, s, ~] = svd(h);

            s = diag(s);            
            v = v(:, s > TOL);
            s = 1./s(s > TOL);

            q{j}.lambda.b(k) = 0.5 * e(:,k)' * e(:,k) + ...
              ... %0.5 * trace(q{j}.theta.pi * v * diag(s) * v') + ...
              ... % Last line should be working but it's not
                ptheta.p.lambda.b(k);
        end
    end
end

for j = 1:nt 
    theta{j}.lambda = log(q{j}.lambda.a) - log(q{j}.lambda.b);
end
ntheta = theta;
end

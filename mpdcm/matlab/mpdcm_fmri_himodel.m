function [q] = mpdcm_fmri_himodel(dcm, optheta, pars)
%% Model inversion of a hierarchical model 
% 
% Input:
% dcm -- Cell array of dcms
% pars -- Parameters for the inversion.
%
% Output:
% q -- Sufficient statistics of the variational energies.
%
% Defaults
% pars.niter -- Number of iterations of the algorithm. Defaults to 5.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

    if nargin < 3
        pars = struct();
    end

    pars = config_pars(pars);

    [y, u, theta, ptheta] = mpdcm_fmri_tinput(dcm);
    [q, theta, ptheta] = initilize_parameters(u, theta, ptheta, optheta);

    for i = 1:pars.niter
        [q] = update_first_level(q, y, u, theta, ptheta);
        [q] = update_second_level(q, y, u, theta, ptheta);
    end

end

function [ q ] = update_first_level(q, y, u, theta, ptheta)
% Update theta

    nt = numel(theta);
    np = size(ptheta.p.eta.mu, 1);
    nr = theta{1}.dim_x;
    ns = numel(y{1})/nr;

    % Use E[p(y | theta_i, lambda_i)]_q(\lambda_i)

    mu = cell(nt, 1);
    for j = 1:nt
        tlambda = full(log(q.lambda(j).a) - log(q.lambda(j).b));
        mu{j} = [q.theta(j).mu; tlambda];
    end

    theta = mpdcm_fmri_set_parameters(mu, theta, ptheta);

    %% Use E[p(theta_i|eta, rho)]_{q(rho) q(eta)} for the conditional 
    % distribution

    tptheta = ptheta;
    tptheta.p.theta = struct('mu', [], 'pi', []);
    tptheta.p.theta.mu = (ptheta.p.x * q.eta.mu')';

    % Non spheric covariance matrix

    tptheta.p.theta.pi = ...
        blkdiag(...
        (q.rho.a(1)/q.rho.b(1))*ptheta.h.theta.pi(1:end-nr*2-1,1:end-nr*2-1),...
        (q.rho.a(2)/q.rho.b(2))*ptheta.h.theta.pi(end-nr*2:end,end-nr*2:end));

    %% Conditional MAP estimator

    tptheta.p.theta.mu = cat(1, tptheta.p.theta.mu, zeros(nr, nt));
    tptheta.p.theta.pi = blkdiag(tptheta.p.theta.pi, eye(nr));

    [mu, ny, dfdx] = mpdcm_fmri_map(y, u, theta, tptheta);
    theta = mpdcm_fmri_set_parameters(mu, theta, tptheta);

    %% Compute the update of q(lambda_i)

    for j = 1:nt

        q.theta(j).mu = mu{j}(1:np);

        % Approximate hessian
        % Weight the hessian by the precisions
        dfdx{j} = dfdx{j}(:,:,1:np);

        h1 = bsxfun(@times, dfdx{j}, reshape(exp(theta{j}.lambda), 1, nr, 1));
        h1 = reshape(h1, ns*nr, 1, np);
        h1 = squeeze(h1);

        h2 = reshape(dfdx{1}, ns*nr, 1, np);
        h2 = squeeze(h2);

        q.theta(j).pi = h1'*h2;

        q.lambda(j).a = 0.5*ns + ptheta.p.lambda(j).a - 1;

        e = ny{j} - y{j}';

        % Region specific lambda_i
        for k = 1:nr
            h = dfdx{1}(:, k, :);
            h = squeeze(h);
            q.lambda(j).b(k) = 0.5*e(:,k)'*e(:,k) + ...
               0.5*trace(q.theta(j).pi\(h'*h)) + ptheta.p.lambda(j).b(k);                end
    end
end


function [q] = update_second_level(q, y, u, theta, ptheta)

    nt = numel(theta);
    nr = theta{1}.dim_x;
    np = size(ptheta.p.eta.mu, 1);

    n = np-(nr*2+1);
    m = nr*2+1;

    p = ptheta.p;

    % Big theta
    btheta = cell2mat({q.theta.mu})';

    % MLE 
    % We don't need to solve any linear system here
    eta = p.x'*btheta; % p.kx\(p.x'*btheta)
    % MAP
    eta = (p.eta.d + p.kx)\(p.eta.d*p.eta.mu' + eta);

    q.eta.mu = eta';
    q.eta.d = (p.eta.d + p.kx);

    q.rho.a = 0.5 * nt * [n; m] + p.rho.a - 1;
 
    p1 = ptheta.h.theta.pi(1:n, 1:n);
    p2 = ptheta.h.theta.pi(n+1:end, n+1:end);

    q.rho.b = p.rho.b;

    phi = btheta' * btheta - (eta' * p.kx * eta)';

    for i = 1:nt
        phi = phi + inv(q.theta(i).pi);
    end
    phi = ptheta.h.theta.pi * phi;
    phi = diag(phi);

    q.rho.b = p.rho.b + 0.5 * [sum(phi(1:n)); sum(phi(n+1:end))];

end

function [q, theta, ptheta] = initilize_parameters(u, theta, ptheta, optheta)
%% Initilizes the parameters with the required fields

    nu = size(u{1}, 2);
    nt = numel(theta);
    nr = theta{1}.dim_x;
    np = numel(ptheta.p.theta.mu);
    nx = size(optheta.x, 2);

    p = struct('lambda', [], 'theta', [], 'rho', [],'eta', []);
    p.lambda = struct('a', cell(nt, 1), 'b', []);
    p.theta = struct('mu', cell(nt, 1), 'pi', [], 'sigma', [], 'chol_pi', []);
    p.rho = struct('a', [], 'b', []);
    p.eta = struct('mu', [], 'pi', [], 'chol_pi', [], 'sigma', []);

    p.rho.a = nr*np*nu/3*ones(2, 1);
    p.rho.b = 2*ones(2, 1);

    p.eta.mu = [ptheta.p.theta.mu(1:end-nr) zeros(np-nr, nx-1)];
    p.eta.d = eye(nx);
    p.eta.pi = ptheta.p.theta.pi(1:end-nr, 1:end-nr);
    p.eta.chol_pi = ptheta.p.theta.chol_pi;
    p.eta.sigma = ptheta.p.theta.sigma;
   
    % Set regressor matrix

    p.x = optheta.x;
    p.kx = p.x'*p.x;

    % Moment generating function of a gaussian, equivalent to the mean and the
    % second moment of a log normal

    e_lambda = ptheta.p.theta.mu(end-nr+1:end);
    c_lambda = diag(ptheta.p.theta.sigma);
    c_lambda = c_lambda(end-nr+1:end);
    c_lambda = c_lambda(:);

    m1 = exp(e_lambda + 0.5*c_lambda);
    m2 = exp(2*e_lambda + 0.5*c_lambda*4);

    b = 1./((m2./m1) - m1);
    a = m1.*b;

    for i = 1:nt
        p.lambda(i).a = a;
        p.lambda(i).b = b;
    end

    % Sufficient statistics for the variational energies

    q = p;

    q.rho.a = nt*ones(2, 1);
    q.rho.b = nt*ones(2, 1);

    pmu = p.eta.mu * p.x';

    for i = 1:nt
        q.theta(i).mu = pmu(:, i); 
        q.theta(i).pi = p.eta.pi;
        q.theta(i).chol_pi = p.eta.chol_pi;
        q.theta(i).sigma = p.eta.sigma;
    end

    % Update the priors

    % Prior covariance of the parameters theta. Parameters are not close to 
    % spheric. Because the derivatives seem to have a difference of 10^1, and
    % the Hessian is approximated with the J'J the difference should be 
    % something like 10^2 

    h.theta = struct('pi', []);
    h.theta.pi = p.eta.pi;

    % Set the regressor matrix

    ptheta.p = p;
    ptheta.h = h;

end

function [pars] = config_pars(pars)
%% Default values

    if ~isfield(pars, 'niter')
        pars.niter = 10;
    end

end

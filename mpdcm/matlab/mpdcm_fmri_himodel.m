function [theta, htheta] = mpdcm_fmri_himodel(dcm, pars)
%% Model inversion of a hierarchical model 
% 
% Input:
% dcm -- Cell array of dcms
% pars -- Parameters for the inversion.
%
% Output:
% theta -- Sufficient statistics of the individual record distributions.
% htheta -- Sufficient statistics of the hyper parameters.
%
% Defaults
% pars.niter -- Number of iterations of the algorithm. Defaults to 5.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

    if nargin < 2
        pars = struct();
    end

    pars = config_pars(pars);

    [y, u, theta, ptheta] = mpdcm_fmri_tinput(dcm);
    [q, theta, ptheta] = initilize_parameters(theta, ptheta);

    for i = 1:pars.niter
        [q] = update_first_level(q, y, u, theta, ptheta);
        [q] = update_q_h_theta(q, y, u, theta, ptheta);
        [q] = update_q_h_theta(q, y, u, theta, ptheta);
    end

end


function [ q ] = update_first_level(p, q, y, u, theta, ptheta)
% Update theta

    nt = numel(theta);
    np = numel(ptheta.p.theta.mu);
    nr = theta{1}.dim_x;
    ns = numel(y{1})/nr;

    % Use E[p(y | theta_i, lambda_i)]_q(\lambda_i)

    for j = 1:nt
        theta{j}.lambda = full(log(q.lambda(i).a) - log(q.lambda(i).b));
    end

    %% Use E[p(theta_i|mu, Lambda)]_{q(mu) (Lambda)} for the conditional 
    % distribution

    tptheta = ptheta;
    tptheta.p.theta.mu = q.eta.mu;
    tptheta.p.theta.pi = q.rho.a/q.rho.b * eye(size(ptheta.p.theta.pi));

    %% Conditional MAP estimator

    [mu, ny, dfdx] = mpdcm_fmri_map(y, u, theta, tptheta);
    theta = mpdcm_fmri_set_parameters(mu, theta, ptheta);

    %% Compute the update of q(lambda_i)

    for j = 1:nt

        q.theta(j).mu = mu{j};

        % Approximate hessian
        % Weight the hessian by the precisions
        dfdx{j} = dfdx{j}(:,:,1:np-nr);

        h1 = bsxfun(@times, dfdx{j}, reshape(exp(theta{j}.lambda), 1, nr, 1));
        h1 = reshape(h1, ns*nr, 1, np-nr);
        h1 = squeeze(h1);

        h2 = reshape(dfdx{1}, ns*nr, 1, np-nr);
        h2 = squeeze(h2);

        q.theta(j).pi = 0.5*h1'*h2;

        q.lambda(j).a = 0.5*ns + ptheta.p.lambda.a - 1;

        e = ny{j} - y{j}';
        % Region specific lambda_i
        for k = 1:nr
            h = dfdx{1}(:, k, :);
            h = squeeze(h);
            h = 0.5*h'*h;
            q.lambda(j).b(k) = 0.5*e(:,k)'*e(:,k) + ...
                trace(q.theta(j).pi\h) + ptheta.p.lambda.b(k);
        end
    end

end

function [q] = update_q_rho(q, y, u, theta, htheta, ptheta)

    nt = numel(nt);
    np = numel(ptheta.p.theta.mu);

    q.rho.a = 0.5*nt*np + ptheta.p.rho.a;
    q.rho.b = ptheta.p.rho.b; 

    for i = 1:nt
        e = q.theta(i).mu - q.eta.mu;  
        q.rho.b = q.rho.b + 0.5 * (e'*e + trace(inv(q.theta(i).pi)));
    end

    % The expectation of q(eta) enters as uncertainty in the trace of its
    % variance
    q.rho.b = q.rho.b + 0.5*nt*trace(inv(q.mu.pi));

end


function [q] = update_q_eta(q, y, u, theta, htheta, ptheta)

    nt = numel(nt);

    q.eta.mu = 0*q.eta.mu;

    % Use E[p(theta_i|mu)]_{q(theta_i)}

    for i = 1:nt
        q.eta.mu = q.eta.mu + q.theta(i).mu;
    end

    q.eta.mu = q.eta.mu/nt;

    % Weight by the expected uncertainty
    erho = nt*q.rho.a/q.rho.b;

    q.eta.mu = q.eta.mu * erho + ptheta.p.eta.pi * ptheta.p.eta.mu;
    q.eta.pi = q.eta.mu * eye(size(ptheta.p.eta.pi)) + ptheta.p.eta.pi;

    q.eta.mu = q.eta.pi\q.eta.mu;

end

function [q, theta, ptheta] = initilize_parameters(theta, ptheta)
%% Initilizes the parameters with the required fields


    nt = numel(theta);

    p = struct('lambda', [], 'theta', [], 'rho', [],'eta', []);
    p.lambda = struct('a', cell(nt, 1), 'b', []);
    p.theta = struct('mu', cell(nt, 1), 'pi', [], 'sigma', [], 'chol_pi', []);
    p.rho = struct('a', [], 'b', []);
    p.eta = struct('mu', [], 'pi', [], 'chol_pi', [], 'sigma', []);

    p.rho.a = 2;
    p.rho.b = 2;

    p.eta.mu = ptheta.p.theta.mu;
    p.eta.pi = ptheta.p.theta.pi;
    p.eta.chol_pi = ptheta.p.theta.chol_pi;
    p.eta.sigma = ptheta.p.theta.sigma;

    % Moment generating function of a gaussian, equivalent to the mean and the
    % second moment of a log normal

    nr = theta{1}.dim_x;

    e_lambda = htheta.p.theta.mu(end-nr+1:end);
    c_lambda = diag(htheta.p.theta.sigma);
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

    for i = 1:nt
        q.theta(i).mu = p.eta.mu;
        q.theta(i).pi = p.eta.pi;
        q.theta(i).chol_pi = p.eta.mu.chol_pi;
        q.theta(i).sigma = p.eta.sigma;
    end

    % Update the priors

    ptheta.p = p;

end

function [pars] = config_pars(pars)
%% Default values

    if ~isfield(pars, 'niter')
        pars.niter = 5;
    end

end

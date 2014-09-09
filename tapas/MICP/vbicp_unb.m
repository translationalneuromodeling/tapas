% Variational Bayes algorithm for approximate mixed-effects inference on
% the classification accuracy using the normal-binomial model.
% 
% Usage:
%     q = vicp_unb(ks,ns,...)
% 
% Arguments:
%     ks: 1xSUBJECTS vector of correct trials
%     ns: 1xSUBJECTS vector of trial numbers
% 
% Optional named arguments:
%     'verbose' (default: 0) verbosity (0 or 1)
% 
% Return value:
%     q: struct that contains all posterior moments:
%     .mu_mu:    mean of the posterior population mean effect
%     .eta_mu:   precision of the posterior population mean effect
%     .a_lamba:  shape parameter of the posterior precision population effect
%     .b_lambda: scale parameter of the posterior precision population effect
%     .mu_rho:   1xN vector of means of posterior subject-specific effects
%     .eta_rho:  1xN vector of precisions of posterior subject-specific effects
%
% Note that all above 'effects' represent accuracies in logit space which
% has infinite support. In order to obtain, e.g., a posterior-mean estimate
% of the population accuracy in the conventional [0..1] space, use:
% logitnmean(q.mu_mu,1/sqrt(q.eta_mu))
%
% Literature:
% Brodersen KH et al. (in preparation)
%
% See also:
%     vbicp_simclf_demo

% Kay H. Brodersen, ETH Zurich
% $Id: vbicp_unb.m 19160 2013-03-25 13:18:49Z bkay $
% ------------------------------------------------------------------------------
function q = vbicp_unb(ks,ns,varargin)
    
    % Process varargin
    defaults.verbose = 0;
    args = propval(varargin,defaults);
    
    % Check input
    assert(size(ks,1)==1, 'ks must be a 1 x SUBJECTS row vector');
    assert(size(ns,1)==1, 'ns must be a 1 x SUBJECTS row vector');
    try, verbose; catch; verbose = 0; end
    assert(verbose==0 | verbose==1, 'invalid verbosity level');
    
    % Set data
    data.ks = ks;
    data.ns = ns;
    data.m = size(data.ks,2);
    
    % Specify prior
    % Prior as of 21/02/2012
    prior.mu_0 = 0;
    prior.eta_0 = 1;
    prior.a_0 = 1;
    prior.b_0 = 1;
    
    % Initialize posterior
    q.mu_mu    = prior.mu_0;
    q.eta_mu   = prior.eta_0;
    q.a_lambda = prior.a_0;
    q.b_lambda = prior.b_0;
    q.mu_rho   = zeros(1,data.m);    % hard-coded prior on lower level
    q.eta_rho  = 0.1*ones(1,data.m); % hard-coded prior on lower level
    q.F        = -inf;
    
    % Variational algorithm
    maxIter = 200;
    for i = 1:maxIter
        q_old = q;
        
        % 1st mean-field partition
        q = update_rho(data,prior,q);
        
        % 2nd mean-field partition
        q = update_mu(data,prior,q);
        
        % 3rd mean-field partition
        q = update_lambda(data,prior,q);
        
        % Free energy (q.F)
        q = free_energy(data,prior,q);
        
        % Convergence?
        if abs(q.F - q_old.F) < 1e-3
            break;
        elseif (i == maxIter)
            warning(['vbicp_unb: reached maximum variational iterations ', ...
                '(',num2str(maxIter), ')']);
        end
    end
end

% ------------------------------------------------------------------------------
% Update 1st mean-field partition
function q = update_rho(data,prior,q)
    
    % Gauss-Newton scheme to find the mode
    % Define Jacobian and Hessian
    dI = @(rho) (data.ks-data.ns.*safesigm(rho)) ...
                + q.a_lambda*q.b_lambda*(q.mu_mu*ones(1,data.m)-rho);
    d2I = @(rho) -diag(data.ns.*safesigm(rho).*(1-safesigm(rho))) ...
                 - q.a_lambda*q.b_lambda*eye(data.m);
    
    % Iterate until convergence to find maximum,
    % then update approximate posterior
    maxIter = 10;
    for i=1:maxIter
        old_mu_rho = q.mu_rho;
        
        %q.mu_rho = q.mu_rho - (inv(d2I(q.mu_rho)) * dI(q.mu_rho)')';
        q.mu_rho = q.mu_rho - (d2I(q.mu_rho) \ dI(q.mu_rho)')';
        
        % Convergence?
        if sum((q.mu_rho - old_mu_rho).^2) < 1e-3
            break
        elseif (i == maxIter)
            warning(['vbicp_unb: argmax_rho: reached maximum GN ', ...
                'iterations (',num2str(maxIter), ')']);
        end
    end
    
    % Update precision
    q.eta_rho = diag(-d2I(q.mu_rho))';
end

% ------------------------------------------------------------------------------
% Update 2nd mean-field partition
function q = update_mu(data,prior,q)
    
    % Update approximate posterior
    q.mu_mu = (prior.mu_0*prior.eta_0 + q.a_lambda*q.b_lambda*sum(q.mu_rho)) ...
              / (data.m*q.a_lambda*q.b_lambda + prior.eta_0);
    q.eta_mu = data.m*q.a_lambda*q.b_lambda + prior.eta_0;
end

% ------------------------------------------------------------------------------
% Update 3rd mean-field partition
function q = update_lambda(data,prior,q)
    
    % Update approximate posterior
    q.a_lambda = prior.a_0 + data.m/2;
    q.b_lambda = 1/(1/prior.b_0 + 1/2 ...
                       *sum((q.mu_rho-q.mu_mu).^2 + 1./q.eta_rho + 1/q.eta_mu));
end

% ------------------------------------------------------------------------------
% Log-joint over data and parameters at the variational mode
function L = logjoint(data,prior,q)
    lambda_mode = (q.a_lambda-1)*q.b_lambda;
    L = sum(log(binopdf(data.ks, data.ns, safesigm(q.mu_rho)))) ...
      + sum(log(normpdf(q.mu_rho, q.mu_mu, 1/sqrt(lambda_mode)))) ...
      + log(normpdf(q.mu_mu, prior.mu_0, 1/sqrt(prior.eta_0))) ...
      + log(gampdf(lambda_mode, prior.a_0, prior.b_0));
end

% ------------------------------------------------------------------------------
% Approximation to the free energy
function q = free_energy(data,prior,q)
    q.F = 1/2*(log(prior.eta_0) - log(q.eta_mu)) ...
        - prior.eta_0/2*((q.mu_mu-prior.mu_0)^2 + 1/q.eta_mu) + q.a_lambda ...
        - prior.a_0*log(prior.b_0) + gammaln(q.a_lambda)-gammaln(prior.a_0) ...
        - q.a_lambda*q.b_lambda*(1/prior.b_0 + data.m/(2*q.eta_mu)) ...
        + (prior.a_0 + data.m/2)*log(q.b_lambda) ...
        + (prior.a_0 - q.a_lambda + data.m/2)*digamma(q.a_lambda) + 1/2 ...
        + sum(log(binopdf(data.ks, data.ns, safesigm(q.mu_rho))) ...
              - 1/2*q.a_lambda*q.b_lambda*(q.mu_rho-q.mu_mu).^2 ...
              - 1/2*log(q.eta_rho));
end

% ------------------------------------------------------------------------------
% Digamma function, based on a simple numerical approximation
function v = digamma(x)
    h = 1e-5;
    dgamma = (gamma(x-2*h) -8*gamma(x-h) +8*gamma(x+h) -gamma(x+2*h)) /12 /h;
    v = dgamma./gamma(x);
end

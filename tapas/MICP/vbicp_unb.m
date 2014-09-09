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
%     .mu_rho:   1xN vector of the means of the posterior subject-specific effects
%     .eta_rho:  1xN vector of the precisions of the posterior subject-specific effects
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
% $Id: vbicp_unb.m 17889 2012-10-22 08:25:34Z bkay $
% -------------------------------------------------------------------------
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
    q.mu_mu = prior.mu_0;
    q.eta_mu = prior.eta_0;
    q.a_lambda = prior.a_0;
    q.b_lambda = prior.b_0;
    q.mu_rho = zeros(1,data.m);     % hard-coded prior on lower level
    q.eta_rho = 0.1*ones(1,data.m); % hard-coded prior on lower level
    
    % Begin EM iterations
    maxIter = 50;
    for i=1:maxIter
        old_q = q;
        
        % 1st partition
        q = argmax_rho(data,prior,q);
        
        % 2nd partition
        q = argmax_mu(data,prior,q);
        
        % 3rd partition
        q = argmax_lambda(data,prior,q);
        
        % Convergence?
        tmp_old = [old_q.mu_mu, old_q.eta_mu, old_q.a_lambda, old_q.b_lambda, old_q.mu_rho, old_q.eta_rho];
        tmp_new = [    q.mu_mu,     q.eta_mu,     q.a_lambda,     q.b_lambda,     q.mu_rho,     q.eta_rho];
        if sum((tmp_new - tmp_old).^2) < 1e-3, break; end
        if (i == maxIter), warning(['vbicp_unb: reached maximum EM iterations (',num2str(maxIter), ')']); end
    end
end

% -------------------------------------------------------------------------
% Maximizes the 1st variational energy
function q = argmax_rho(data,prior,q)
    
    % Gauss-Newton scheme to find the mode
    % Define Jacobian and Hessian
    dI = @(rho) (data.ks-data.ns.*safesigm(rho)) + q.a_lambda*q.b_lambda*(q.mu_mu*ones(1,data.m)-rho);
    d2I = @(rho) -diag(data.ns.*safesigm(rho).*(1-safesigm(rho))) - q.a_lambda*q.b_lambda*eye(data.m);
    
    % Iterate until convergence to find maximum,
    % then update approximate posterior
    maxIter = 10;
    for i=1:maxIter
        old_mu_rho = q.mu_rho;
        
        %q.mu_rho = q.mu_rho - (inv(d2I(q.mu_rho)) * dI(q.mu_rho)')';
        q.mu_rho = q.mu_rho - (d2I(q.mu_rho) \ dI(q.mu_rho)')';
        
        % Convergence?
        if sum((q.mu_rho - old_mu_rho).^2) < 1e-3, break; end
        if (i == maxIter), warning(['vbicp_unb: argmax_rho: reached maximum GN iterations (',num2str(maxIter), ')']); end
    end
    
    % Update precision
    q.eta_rho = diag(-d2I(q.mu_rho))';
end

% -------------------------------------------------------------------------
% Maximizes the 2nd variational energy
function q = argmax_mu(data,prior,q)
    
    % Update approximate posterior
    q.mu_mu = (prior.mu_0*prior.eta_0 + q.a_lambda*q.b_lambda*sum(q.mu_rho)) ...
              / (data.m*q.a_lambda*q.b_lambda + prior.eta_0);
    q.eta_mu = data.m*q.a_lambda*q.b_lambda + prior.eta_0;
end

% -------------------------------------------------------------------------
% Maximizes the 3rd variational energy
function q = argmax_lambda(data,prior,q)
    
    % Update approximate posterior
    q.a_lambda = prior.a_0 + data.m/2;   % TODO: may want to double-check that this is constant
    q.b_lambda = 1/(1/prior.b_0 + 1/2*sum((q.mu_rho-q.mu_mu).^2 + 1./q.eta_rho + 1/q.eta_mu));
end

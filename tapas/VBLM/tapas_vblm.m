% Variational Bayesian multiple linear regression.
%
% Basic usage:
%     q = tapas_vblm(y, X)
%
% Full usage:
%     [q, stats, q_trace] = tapas_vblm(y, X, a_0, b_0, c_0, d_0)
%
% Args:
%     y:   <n x 1> vector of observations (response variable)
%     X:   <n x d> design matrix (regressors)
%     a_0: shape parameter of the prior precision of coefficients
%     b_0: rate  parameter of the prior precision of coefficients
%     c_0: shape parameter of the prior noise precision
%     d_0: rate  parameter of the prior noise precision
%
% Returns:
%     q: moments of the variational posterior:
%       .mu_n:     posterior expectation of coefficients
%       .Lambda_n: posterior precision matrix of coefficients
%       .a_n:      shape parameter of the posterior precision of coefficients
%       .b_n:      rate  parameter of the posterior precision of coefficients
%       .c_n:      shape parameter of the posterior noise precision
%       .d_n:      rate  parameter of the posterior noise precision
%       .F:        free energy of the model given the data
%       .prior:    struct with a copy of the prior moments a_0, b_0, c_0, d_0
%     stats: additional statistics:
%       .logBF_beta: log Bayes factor between the full model and a reduced model
%                    in which one regressor at a time has been omitted
%       .logBF_null: log Bayes factor between the full model and a null model
%                    whose only regressor is a constant
%     q_trace: all intermediate results
%
% The regression model is
%     p(alpha)            = Ga(alpha | a_0, b_0)
%     p(beta | alpha)     = N(beta | 0, alpha^-1 I)
%     p(lambda)           = Ga(lambda | c_0, d_0)
%     p(y | beta, lambda) = Prod_i N(y_i | beta^T*x_i, lambda^-1)
% where
%     y:      data (response variable)
%     X:      design matrix (regressors)
%     beta:   coefficients
%     alpha:  precision of coefficients
%     lambda: precision of observation noise
%
% The model is inverted by optimizing a variational approximate posterior
% q(alpha, beta, lambda) ~ p(alpha, beta, lambda | y, X) with respect to the
% (negative) free energy, which itself is a lower bound to the log model
% evidence. The variational posterior is derived under the mean-field
% assumption, i.e., q(alpha, beta, lambda) = q(alpha)*q(beta)*q(lambda).
%
% See also:
%     tapas_vblm_predict

% Kay H. Brodersen, TNU, ETH Zurich
% $Id: tapas_vblm.m 19999 2013-10-05 21:26:13Z bkay $
% ------------------------------------------------------------------------------
function [q, stats, q_trace] = tapas_vblm(y, X, a_0, b_0, c_0, d_0)

    % Default prior
    if nargin <= 2, warning('Using the default prior is discouraged'); end
    try; a_0; catch; a_0 = 2;   end
    try; b_0; catch; b_0 = 0.2; end
    try; c_0; catch; c_0 = 10;  end
    try; d_0; catch; d_0 = 1;   end
    prior.a_0 = a_0; prior.b_0 = b_0; clear a_0 b_0
    prior.c_0 = c_0; prior.d_0 = d_0; clear c_0 d_0
    
    % Check input
    y = check_input(y,X,prior);
    
    % Invert full model
    if nargout >= 3
        [q, q_trace] = invert_model(y, X, prior);
    else
        q            = invert_model(y, X, prior);
    end
    
    % Additional statistics?
    if nargout >= 2
        % Compare with reduced models (omitting one regressor a time)
        for i = 1:size(X,2)
            tmp_X = X(:, [1:(i-1),(i+1):end]);
            tmp_q = invert_model(y, tmp_X, prior);
            stats.logBF_beta(i) = q.F - tmp_q.F;
        end
        
        % Compare with null model (constant regressor only)
        tmp_X = ones(size(X,1),1);
        tmp_q = invert_model(y, tmp_X, prior);
        stats.logBF_null = q.F - tmp_q.F;
    end
    
end

% -----------------------------------------------------------------------------
% Returns the variational posterior q that maximizes the free energy.
function [q, q_trace] = invert_model(y, X, prior)

    % Data shortcuts
    [n,d] = size(X); % observations x regressors
    
    % Initialize variational posterior
    q.mu_n     = zeros(d,1);
    q.Lambda_n = eye(d);
    q.a_n      = prior.a_0;
    q.b_n      = prior.b_0;
    q.c_n      = prior.c_0;
    q.d_n      = prior.d_0;
    q.F        = -inf;
    q.prior    = prior;
    
    % Initialize trace of intermediate results?
    if nargout >= 2
        q_trace(1).q = q;
        q_trace(1).q.F = free_energy(q,y,X,prior);
    end
    
    % Variational algorithm
    nMaxIter = 30;
    kX = X'*X;
    for i = 1:nMaxIter
        
        % (1) Update q(beta)
        q.Lambda_n = q.a_n/q.b_n + q.c_n/q.d_n * ((X')*X);
        q.mu_n = q.c_n/q.d_n * (q.Lambda_n \ ((X')*y));
        
        % (2) Update q(alpha)
        q.a_n = prior.a_0 + d/2;
        q.b_n = prior.b_0 + 1/2 * (q.mu_n'*q.mu_n + trace(inv(q.Lambda_n)));
        
        % (3) Update q(lambda)
        q.c_n = prior.c_0 + n/2;
        pe = y - X*q.mu_n;
        q.d_n = prior.d_0 + 0.5 * (pe'*pe + trace(q.Lambda_n\kX)) ;
        
        % Compute free energy
        F_old = q.F;
        q.F = free_energy(q,y,X,prior);
        
        % Append to trace of intermediate results?
        if nargout >= 2, q_trace(i+1).q = q; end
        
        % Convergence?
        if (q.F - F_old < 10e-4), break; end
        if (i == nMaxIter), warning('tapas_vblm: reached %d iterations',nMaxIter); end
    end
end

% ------------------------------------------------------------------------------
% Computes the free energy of the model given the data.
function F = free_energy(q,y,X,prior)
    % Data shortcuts
    n = size(X, 1);
    d = size(X, 2);
    
    % Expected log joint <ln p(y,beta,alpha,lambda)>_q
    J = n/2*(digamma(q.c_n)-log(q.d_n)) - n/2*log(2*pi) ...
        - 0.5*q.c_n/q.d_n*((y')*y) + q.c_n/q.d_n*((q.mu_n')*(X')*y) ...
        - 0.5*q.c_n/q.d_n*trace((X')*X * (q.mu_n*q.mu_n' + inv(q.Lambda_n))) ...
      - d/2*log(2*pi) + n/2*(digamma(q.a_n)-log(q.b_n)) ...
        - 0.5*q.a_n/q.b_n * (q.mu_n'*q.mu_n + trace(inv(q.Lambda_n))) ...
      + prior.a_0*log(prior.b_0) - gammaln(prior.a_0) ...
        + (prior.a_0-1)*(digamma(q.a_n)-log(q.b_n)) - prior.b_0*q.a_n/q.b_n ...
      + prior.c_0*log(prior.d_0) - gammaln(prior.c_0) ...
        + (prior.c_0-1)*(digamma(q.c_n)-log(q.d_n)) - prior.d_0*q.c_n/q.d_n;
    
    % Entropy H[q]
    H = d/2*(1+log(2*pi)) + 1/2*log(det(inv(q.Lambda_n))) ...
      + q.a_n - log(q.b_n) + gammaln(q.a_n) + (1-q.a_n)*digamma(q.a_n) ...
      + q.c_n - log(q.d_n) + gammaln(q.c_n) + (1-q.c_n)*digamma(q.c_n);

    % Free energy
    F = J + H;
end

% ------------------------------------------------------------------------------
% Checks input validity.
function y = check_input(y,X,prior)
    assert(isvector(y), 'y must be a vector');
    y = y(:);
    assert(length(y)==size(X, 1), 'y and X must have same number of rows');
    assert(ndims(X) == 2, 'X must be a vector or a matrix');
    assert(isscalar(prior.a_0), 'a_0 must be a scalar');
    assert(isscalar(prior.b_0), 'b_0 must be a scalar');
    assert(isscalar(prior.c_0), 'c_0 must be a scalar');
    assert(isscalar(prior.d_0), 'd_0 must be a scalar');
    assert(prior.a_0 > 0, 'a_0 must be positive');
    assert(prior.b_0 > 0, 'b_0 must be positive');
    assert(prior.c_0 > 0, 'c_0 must be positive');
    assert(prior.d_0 > 0, 'd_0 must be positive');
end

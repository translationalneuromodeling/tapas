% Posterior predictive density of a variational Bayesian multiple linear
% regression model.
%
% Usage:
%     [m_new, y_new] = tapas_vblm_predict(X_new, q)
%
% Args:
%     X_new: <n x d> matrix of new regressors
%     q:     variational posterior, as returned by vblm()
%
% Returns:
%     m_new: <n x 1> vector of posterior predictive response mean
%     t_new: <n x 1> vector of posterior predictive response precision

% Kay H. Brodersen, TNU, ETH Zurich
% $Id: tapas_vblm_predict.m 19999 2013-10-05 21:26:13Z bkay $
% ------------------------------------------------------------------------------
function [m_new,t_new] = tapas_vblm_predict(X_new, q)

    % Check input
    assert(ndims(X_new) == 2, 'X_new must be a vector or a matrix');
    assert(size(X_new,2) == length(q.mu_n), ['X_new must have same number ', ...
        'regressors as original design matrix X used in vblm()']);
    assert(isstruct(q), 'q must be a struct returned by vblm()');
    
    % TODO: marginalize lambda
    
    % Evaluate predictive density
    % m_new: posterior predictive mean
    % t_new: posterior predictive precision
    m_new = X_new*q.mu_n;
    t_new = ((q.c_n/q.d_n)^-1 + diag((X_new/q.Lambda_n)*(X_new'))) .^-1;
    
end

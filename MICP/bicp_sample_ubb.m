% Metropolis algorithm to sample from the posterior p(alpha,beta|ks),
% from the posteriors p(pi_j|ks) for each j, and from the posterior
% predictive density p(pi~|ks). Based on the beta-binomial model.
% 
% Usage:
%     [alphas,betas] = bicp_sample_ubb(ks,ns)
%     [alphas,betas,pijs] = bicp_sample_ubb(ks,ns)
%     [alphas,betas,pijs,pis] = bicp_sample_ubb(ks,ns)
%     [alphas,betas,pijs,pis] = bicp_sample_ubb(ks,ns,nSamples,burnIn)
%     [...,mixing] = bicp_sample_ubb(ks,ns)
% 
% Arguments:
%     ks: 1 x SUBJECTS vector specifying the number of correctly predicted
%         trials in each subject
%     ns: 1 x SUBJECTS vector specifying the total number of trials
%         in each subject
%     nSamples (default: 10000): number of samples to return
%     burnIn (default: 20% of nSamples, but at least 100): either an
%         absolute number of steps to initialize the Markov chain (e.g.
%         1000), or a fraction with respect to nSamples (e.g. 0.1)
% 
% Return values:
%     alphas, betas: samples from the posterior p(alpha,beta|ks)
%     pijs: samples from the subject-specific posteriors of pi_j
%     pis: samples from the posterior predictive density p(pi~|ks)
%     mixing: contains Metropolis acceptance rates (for debugging purposes)
% 
% See also:
%     bicp_acc_demo
%     bicp_bacc_bb_demo

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_sample_ubb.m 16246 2012-05-31 08:25:19Z bkay $
% -------------------------------------------------------------------------
function [alphas,betas,pijs,pis,mixing] = bicp_sample_ubb(ks,ns,nSamples,burnIn)
    
    % Check input
    assert(size(ks,1)==1, 'ks must be a 1 x SUBJECTS row vector');
    assert(size(ns,1)==1, 'ns must be a 1 x SUBJECTS row vector');
    try, nSamples; catch; nSamples = 10000; end
    assert(logical(isint(nSamples)),'nSamples must be an integer');
    assert(nSamples>0,'nSamples must be greater than 0');
    try, burnIn; catch; burnIn = max([100,round(0.2*nSamples)]); end
    if isint(burnIn)
        assert(burnIn>1,'illegal value for burnIn');
    elseif (0<burnIn && burnIn<1)
        burnIn = round(burnIn*nSamples);
    else
        error('illegal value for burnIn');
    end
    try, verbose; catch; verbose = 0; end
    assert(verbose==0 | verbose==1, 'invalid verbosity level');
    m = size(ks,2);
    
    % Specify covariance matrix for the proposal distribution q(omega*|omega)
    Sigma_q = [1/8,0;0,1/8];
    
    % Draw initial values (overdispersed sampling)
    omega = orig2trans(abs(normrnd(0,10)),abs(normrnd(0,10)));
    % Kay new idea: omega = normrnd(0,8,1,2);
    pij = betarnd(1,1,m,1);
    
    % Initialize return values
    omegas = NaN(2,nSamples); % transformed alpha and beta
    pijs = NaN(m,nSamples); % pi_j
    pis = NaN(1,nSamples);  % pi~
    mixing = zeros(1+m,1);
    
    % Begin sampling (Metropolis algorithm)
    for t = 1-burnIn:nSamples
        
        % -----------------------------------------------------------------
        % Step 1: sample from the joint p(alpha,beta|ks,ns)
        
        % Create candidate. The proposal distribution is symmetric, so our
        % acceptance probability term is a simple Metropolis step.
        omega_star = mvnrnd(omega,Sigma_q);
        
        % Compute acceptance probability
        log_r = loglikelihood(omega_star,ks,ns) + logprior(omega_star) ...
              - loglikelihood(omega,ks,ns) - logprior(omega);
        a = min([1,exp(log_r)]);
        
        % Accept transition?
        if rand < a
            omega = omega_star;
            if t>0, mixing(1) = mixing(1) + 1; end
        end
        
        % Store current value
        if t>0, omegas(:,t) = omega; end
        
        % For the steps below, also remember current omega in alpha,beta
        % parameterization
        [alpha beta] = trans2orig(omega);
        
        
        % -----------------------------------------------------------------
        % Step 2 [fork]: sample from p(pi_j|k), to obtain a posterior
        % density for pi_j for each j. (This step is skipped above, since
        % we are doing Metropolis-Hastings sampling and not Gibbs
        % sampling.)
        if nargout>=3 && t>0
            pijs(:,t) = betarnd(alpha+ks,beta+ns-ks)';
        end
        
        % -----------------------------------------------------------------
        % Step 3 [fork]: sample from the posterior predictive density
        % p(pi~|alpha,beta), where alpha,beta represent the current sample
        % drawn above.
        if nargout>=4
            if t>0
                pis(t) = betarnd(alpha,beta);
            end
        end
        
    end % t
    
    % Convert to original parameterization
    [alphas,betas] = trans2orig(omegas');
    alphas = alphas';
    betas = betas';
    
    % Finalize mixing information
    mixing = mixing/nSamples;
end


% -------------------------------------------------------------------------
% Computes ln p~(omega), the unnormalized log prior on omega.
function lp = logprior(omega)
    [alpha beta] = trans2orig(omega);
    % Gelman et al. 2003, p. 128, suggests gamma=0, which is improper.
    % Here, we use gamma=1, which is proper.
    lp = (-5/2)*log(alpha+beta+1)+log(alpha)+log(beta);
end

% -------------------------------------------------------------------------
% Computes ln p(ks,omega), the log likelihood for omega given the data.
% The 'n choose k' factor is omitted, as it would cancel out in the
% acceptance test.
function ll = loglikelihood(omega,ks,ns)
    [alpha beta] = trans2orig(omega);
    ll = sum( gammaln(alpha+beta)+gammaln(alpha+ks)+gammaln(beta+ns-ks)...
        -gammaln(alpha)-gammaln(beta)-gammaln(alpha+beta+ns) );
end

% -------------------------------------------------------------------------
% Converts hyperparameters into their transformed form.
% See Gelman et al. 2003, p. 128.
function omega = orig2trans(alpha,beta)
    omega(1) = log(alpha/beta);
    omega(2) = log(alpha+beta);
end

% -------------------------------------------------------------------------
% Converts transformed hyperparameters back into their original form.
% See Gelman et al. 2003, p. 128.
function [alpha beta] = trans2orig(omega)
    x = exp(omega(:,1));
    y = exp(omega(:,2));
    alpha = y.*x./(1+x);
    beta  = y./(1+x);
end

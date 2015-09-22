% Gibbs sampling algorithm to sample from the posterior p(mu,lambda|ks),
% from the posteriors p(pi_j|ks), and from the posterior predictive density
% p(pi~|ks,ns). Based on the univariate normal-binomial model with
% independent priors on the population mean and precision (i.e., the same
% model as in vbicp_unb.m).
% 
% Usage:
%     [mus,lambdas,pijs,pis] = bicp_sample_unb(ks,ns,nSamples,burnIn,verbose)
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
%     mus: samples from the posterior population mean
%     lambdas: samples from the posterior population precision (inverse variance)
%     pijs: samples from the subject-specific posteriors of pi_j
%     pis: samples from the posterior predictive density p(pi~|ks)
%     mixing: contains Metropolis acceptance rates (for debugging purposes)
%
% Note that all return values are given logit space.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_sample_unb.m 16210 2012-05-31 07:04:48Z bkay $
% -------------------------------------------------------------------------
function [mus,lambdas,pijs,pis,mixing] = bicp_sample_unb(ks,ns,nSamples,burnIn)
    
    % Check input
    assert(size(ks,1)==1, 'ks must be a 1 x SUBJECTS row vector');
    assert(size(ns,1)==1, 'ns must be a 1 x SUBJECTS row vector');
    assert(size(ks,2)==size(ns,2), 'ks and ns must have the same number of columns');
    try, nSamples; catch; nSamples = 10000; end
    assert(logical(isint(nSamples)),'nSamples must be an integer');
    assert(nSamples>0,'nSamples must be greater than 0');
    try, burnIn; catch; burnIn = max([100,round(0.2*nSamples)]); end
    if isint(burnIn)
        assert(burnIn>1,'illegal value for burnIn');
    elseif (0<burnIn && burnIn<1)
        burnIn = round(burnIn*nSamples);
    else
        error('invalid value for burnIn');
    end
    m = size(ks,2);
    
    % Set data
    data.ks = ks;
    data.ns = ns;
    data.m = m;    
    clear ks ns m
    
    % Specify prior
    % Prior as of 21/02/2012
    prior.mu_0 = 0;
    prior.eta_0 = 1;
    prior.a_0 = 1;
    prior.b_0 = 1;
    
    % Draw initial values (overdispersed sampling)
    mu = normrnd(0,3);
    lambda = gamrnd(1,1/10);
    rhoj = normrnd(0,3,1,data.m);
    
    % Prepare vectors for collecting samples
    mus = NaN(1,nSamples);
    lambdas = NaN(1,nSamples);
    rhojs = NaN(data.m,nSamples);
    pijs = NaN(data.m,nSamples);
    pis = NaN(1,nSamples);
    mixing = 0;
    
    % Begin sampling
    for t = 1-burnIn:nSamples
        
        % -----------------------------------------------------------------
        % Step 1: find a new mu by sampling from p(mu|...)
        % see Gelman, p.49 (2.12)
        mu_m = (prior.eta_0*prior.mu_0 + data.m*lambda*mean(rhoj)) / (prior.eta_0 + data.m*lambda);
        eta_m = prior.eta_0 + data.m*lambda;        
        mu = normrnd(mu_m, 1/sqrt(eta_m));
        
        % -----------------------------------------------------------------
        % Step 2: find a new lambda by sampling from p(lambda|...)
        % see Bishop, p.100
        % Note Bishop uses (alpha,beta) parameterization, calling it (a,b), mean = a/b
        % whereas Matlab uses (k,theta) parameterization, calling it (a,b), mean = a*b
        a_m = prior.a_0 + data.m/2;
        b_m = prior.b_0 + 1/2*sum((rhoj-mu).^2);  % Bishop's parameterization
        lambda = gamrnd(a_m, 1/b_m);              % Converted to Matlab
        
        for j=1:data.m
            % -------------------------------------------------------------
            % Step 3: find a new rho_j by sampling from p(rho_j|...)
            % This can be implemented using a simple Metropolis step.
            rho_j_star = normrnd(rhoj(j),0.8); %%%
            r = log(binopdf(data.ks(j),data.ns(j),sigm(rho_j_star))) ...
              + log(normpdf(rho_j_star,mu,1/sqrt(lambda))) ...
              - log(binopdf(data.ks(j),data.ns(j),sigm(rhoj(j)))) ...
              - log(normpdf(rhoj(j),mu,1/sqrt(lambda)));
            if rand < min([1,exp(r)])
                rhoj(j) = rho_j_star;
                if t>0, mixing = mixing + 1; end
            end
        end % j
        
        % -----------------------------------------------------------------
        % Store current sample
        if t>0
            lambdas(t) = lambda;
            mus(t) = mu;
            rhojs(:,t) = rhoj;
        end
        
        % -------------------------------------------------------------
        % Step 3 [fork]: find sigmoid-transformed posterior samples from
        % p(pi_j|k)
        if nargout>=3
            pij = sigm(rhoj);
            if t>0, pijs(:,t) = pij; end
        end
            
        % -----------------------------------------------------------------
        % Step 4 [fork]: find a new pi~ by ancestral-sampling from
        % p(pi~|mu,Sigma)
        if nargout>=4
            pi = sigm(normrnd(mu,1/sqrt(lambda)));
            if t>0, pis(t) = pi; end
        end
        
    end % t
    
    % Finalize mixing information
    mixing = mixing / (nSamples*data.m);
    
end

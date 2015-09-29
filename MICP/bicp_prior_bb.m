% Returns samples from the prior for the beta-binomial model.
% 
% Usage:
%     [alphas,betas,pis] = bicp_prior_bb(nSamples)
%     [alphas,betas,pis] = bicp_prior_bb(nSamples,burnIn)
% 
% Arguments:
%     nSamples (default: 10000): number of samples to return
%     burnIn (default: 20% of nSamples, but at least 100): either an
%         absolute number of steps to initialize the Markov chain (e.g.
%         1000), or a fraction with respect to nSamples (e.g. 0.1)

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_prior_bb.m 13694 2012-01-05 09:44:08Z bkay $
% -------------------------------------------------------------------------
function [alphas,betas,pis,mixing] = bicp_prior_bb(nSamples,burnIn)
    
    % Initialization
    try, nSamples; catch; nSamples = 10000; end
    assert(logical(isint(nSamples)),'nSamples must be an integer');
    assert(nSamples>0,'nSamples must be greater than 0');
    if ~exist('burnIn','var') || isempty(burnIn), burnIn = max([100,round(0.2*nSamples)]); end
    if isint(burnIn)
        assert(burnIn>1,'illegal value for burnIn');
    elseif (0<burnIn && burnIn<1)
        burnIn = round(burnIn*nSamples);
    else
        error('invalid value for burnIn');
    end
    
    %Sigma_q = [1/32,0;0,1/32];
    Sigma_q = [5,0;0,5];
    omega = orig2trans(abs(normrnd(0,10)),abs(normrnd(0,10)));
    omegas = NaN(nSamples,2);
    lmes = NaN(nSamples,1);
    mixing = 0;
    
    for t = 1-burnIn:nSamples
        % Sample from the joint prior p(alpha,beta|M)
        omega_star = mvnrnd(omega,Sigma_q);
        log_r = logprior(omega_star) - logprior(omega);
        a = min([1,exp(log_r)]);
        if rand < a
            omega = omega_star;
            if t>0, mixing = mixing + 1; end
        end
        if t>0
            omegas(t,:) = omega;
        end
    end
    
    % Return alphas,betas
    [alphas,betas] = trans2orig(omegas);
    
    % Ancestral-sample from the prior p(pi|M)
    pis = betarnd(alphas,betas);
    
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
% Converts hyperparameters into their transformed form.
function omega = orig2trans(alpha,beta)
    omega(1) = log(alpha/beta);
    omega(2) = log(alpha+beta);
end

% -------------------------------------------------------------------------
% Converts transformed hyperparameters back into their original form.
function [alpha beta] = trans2orig(omega)
    x = exp(omega(:,1));
    y = exp(omega(:,2));
    alpha = y.*x./(1+x);
    beta  = y./(1+x);
end

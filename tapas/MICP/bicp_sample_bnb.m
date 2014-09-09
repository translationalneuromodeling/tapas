% Gibbs sampling algorithm to sample from the posterior p(mu,Sigma|ks),
% from the posteriors p(pi_j|ks), and from the posterior predictive density
% p(pi~|ks,ns). Based on the normal-binomial model.
%
% Usage:
%     [mus,Sigmas,pijs,pis] = bicp_sample_bnb(ks,ns,nSamples,burnIn)
% 
% Arguments:
%     ks: 2 x SUBJECTS vector specifying the number of correctly predicted
%         trials in each subject
%     ns: 2 x SUBJECTS vector specifying the total number of trials
%         in each subject
%     nSamples (default: 10000): number of samples to return
%     burnIn (default: 20% of nSamples, but at least 100): either an
%         absolute number of steps to initialize the Markov chain (e.g.
%         1000), or a fraction with respect to nSamples (e.g. 0.1)
% 
% Return values:
%     mus, Sigmas: samples from the posterior p(mu,Sigma|ks)
%     pijs: samples from the subject-specific posteriors of pi_j
%     pis: samples from the posterior predictive density p(pi~|ks)
%     mixing: contains Metropolis acceptance rates (for debugging purposes)

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_sample_bnb.m 16174 2012-05-29 12:34:01Z bkay $
% -------------------------------------------------------------------------
function [mus,Sigmas,pijs,pis,mixing] = bicp_sample_bnb(ks,ns,nSamples,burnIn)
    
    % Check input
    assert(size(ks,1)==2, 'ks must be a 2xSUBJECTS matrix');
    assert(size(ns,1)==2, 'ns must be a 2xSUBJECTS matrix');
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
    
    % Strategy: we will sample from the joint distribution
    % p(mu,Sigma,rho_1:m) and then return the samples
    % mus, Sigmas, pijs, and pis.
    
    % Specify prior on p(mu,Sigma)
    % in terms of (mu_0, Lambda_0/kappa_0; df_0,Lambda_0)
    % (20/01/2012)
    mu_0 = [0; 0];
    kappa_0 = 1;
    Lambda_0 = inv([1 0; 0 1]);
    nu_0 = 5;
    
    % Draw initial values (overdispersed sampling)
    mu = normrnd(0,3,2,1);          % e.g. [0;0]
    Sigma = iwishrnd([3 0; 0 3],3); % e.g. [3 0; 0 3]
    rhoj = normrnd(0,3,2,m);        % e.g. zeros(2,m)
    
    % Prepare vectors for collecting samples
    mus = NaN(2,nSamples);
    Sigmas = NaN(2,2,nSamples);
    rhojs = NaN(2,m,nSamples);
    pijs = NaN(2,m,nSamples);
    pis = NaN(2,nSamples);
    mixing = 0;
    
    % Begin sampling
    for t = 1-burnIn:nSamples
        
        % Initialize fork values
        pij = zeros(2,m);
        pi = zeros(2,1);
        
        % -----------------------------------------------------------------
        % Step 1a: find a new Sigma by sampling from p(Sigma|...)
        kappa_m = kappa_0 + m;
        nu_m = nu_0 + m;
        mu_m = kappa_0/kappa_m*mu_0 + m/kappa_m*mean(rhoj,2);
        S = zeros(2,2);
        for j=1:m
            S = S + (rhoj(:,j)-mean(rhoj,2)) * (rhoj(:,j)-mean(rhoj,2))';
        end
        Lambda_m = Lambda_0 + S + kappa_0*m/kappa_m*(mean(rhoj,2)-mu_0)*(mean(rhoj,2)-mu_0)';
        Sigma = iwishrnd(Lambda_m,nu_m);
        % Note Matlab's different parameterization than in Gelman. The
        % inverse of Sigma has a Wishart distribution with covariance
        % matrix inv(Lambda_m).
        
        % -----------------------------------------------------------------
        % Step 1b: find a new mu by sampling from p(mu|...)
        mu = mvnrnd(mu_m,Sigma/kappa_m)';
        
        for j=1:m
            % -------------------------------------------------------------
            % Step 2: find a new rho_j by sampling from p(rho_j|...)
            % This can be implemented using a simple Metropolis step.
            rho_j_star = mvnrnd(rhoj(:,j),[1 0; 0 1])';
            r = log(binopdf(ks(1,j),ns(1,j),sigm(rho_j_star(1)))) ...
              + log(binopdf(ks(2,j),ns(2,j),sigm(rho_j_star(2)))) ...
              + log(mvnpdf(rho_j_star,mu,Sigma)) ...
              - log(binopdf(ks(1,j),ns(1,j),sigm(rhoj(1,j)))) ...
              - log(binopdf(ks(2,j),ns(2,j),sigm(rhoj(2,j)))) ...
              - log(mvnpdf(rhoj(:,j),mu,Sigma));
            if rand < min([1,exp(r)])
                rhoj(:,j) = rho_j_star;
                if t>0, mixing = mixing + 1; end
            end
            
            % -------------------------------------------------------------
            % Step 3 [fork]: find the sigmoid-transformed posterior density
            % p(pi_j|k)
            if nargout>=3
                pij(:,j) = sigm(rhoj(:,j));
            end
            
        end % j
        
        % -----------------------------------------------------------------
        % Step 4 [fork for the predictive distribution]: find a new pi~
        % by ancestral-sampling from p(pi~|mu,Sigma)
        if nargout>=4
            pi = sigm(mvnrnd(mu,Sigma))';
        end
        
        % -----------------------------------------------------------------
        % Store current sample (and fork values)
        if t>0
           Sigmas(:,:,t) = Sigma;
           mus(:,t) = mu;
           rhojs(:,:,t) = rhoj;
           pijs(:,:,t) = pij;
           pis(:,t) = pi;
        end
        
    end % t
    
    % Finalize mixing information
    mixing = mixing / (nSamples*m);
    
end

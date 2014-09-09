% Computes an approximation to the log model evidence (marginal likelihood)
% of the combined normal-binomial model, based on samples from the
% bivariate prior density p(pi|m).
%
% Usage:
%     lme = bicp_evidence_bnb(ks,ns,nSamples,burnIn)
%
% Arguments:
%     ks, ns: number of correct (k) and total (n) trials for each subject
%     nSamples (default: 10000): number of samples to return
%     burnIn (default: 20% of nSamples, but at least 100): either an
%         absolute number of steps to initialize the Markov chain (e.g.
%         1000), or a fraction with respect to nSamples (e.g. 0.1)
%
% Return values:
%     lme: log model evidence when modelling the number of correct
%         predictions individually, i.e., regarding k+ and k- as separate
%         observed variables. This is the natural quantity for the
%         normal-binomial model but it CANNOT be compared to the model
%         evidence of the simple beta-binomial model.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_evidence_bnb.m 18931 2013-02-15 10:14:09Z bkay $
% -------------------------------------------------------------------------
function [lme,tmp] = bicp_evidence_bnb(ks,ns,nSamples)
    
    % Initialization
    m = size(ns,2);
    assert(size(ks,1)==2, 'ks must be a 2 x SUBJECT row vector');
    assert(all(all(ks<=ns)));
    try, nSamples; catch; nSamples = 10000; end
    assert(logical(isint(nSamples)),'nSamples must be an integer');
    assert(nSamples>0,'nSamples must be greater than 0');
    
    % ---------------------------------------------------------------------
    % Part I: collect samples from p(pi|m)
    % After discussion with Jean and Christoph (20/01/2012)
    mu_0 = [0; 0];
    kappa_0 = 1;
    Lambda_0 = inv([1 0; 0 1]);
    nu_0 = 5;

    % OLD PRIOR FROM MANUSCRIPT
    % mu_0 = [0; 0];
    % kappa_0 = 1;
    % Lambda_0 = inv([0.125 0; 0 0.125]);
    % nu_0 = 8;
    
    mus = NaN(2,nSamples);
    Sigmas = NaN(2,2,nSamples);
    %
    parfor t = 1:nSamples
        Sigma = iwishrnd(Lambda_0,nu_0);
        mu = mvnrnd(mu_0,Sigma/kappa_0);
        mus(:,t) = mu(:);
        Sigmas(:,:,t) = Sigma;
    end
    
    % ---------------------------------------------------------------------
    % Part II: approximate log model evidence
    acc = NaN(nSamples,1);
    parfor t=1:nSamples
        pis = sigm(mvnrnd(mus(:,t),Sigmas(:,:,t),m))';
        tmp = exp(nansum(log(binopdf(ks(1,:),ns(1,:),pis(1,:))) + log(binopdf(ks(2,:),ns(2,:),pis(2,:)))));
        acc(t) = tmp;
    end
    lme = -log(nSamples) + log(nansum(acc));
    
    % In case individual values are of interest:
    tmp = acc;
end

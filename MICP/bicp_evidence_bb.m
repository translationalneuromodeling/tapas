% Computes an approximation to the log model evidence (marginal likelihood)
% of the beta-binomial model, by sampling from the prior density p(pi|m).
%
% Usage:
%     lme = bicp_evidence_bb(ks,ns)
%     lme = bicp_evidence_bb(ks,ns,nSamples,burnIn)
% 
% Arguments:
%     ks, ns: number of correct (k) and total (n) trials for each subject
%     nSamples: number of samples to return
%     burnIn: burn-in period
%
% Example 1: compute the l.m.e. for the single beta-binomial model:
%     lme = bicp_evidence_bb(ks,ns);
%
% Example 2: compute the l.m.e. for the twofold beta-binomial model:
%     lme = bicp_evidence_bb(ks(1,:),ns(1,:)) + bicp_evidence_bb(ks(2,:),ns(2,:));

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_evidence_bb.m 16174 2012-05-29 12:34:01Z bkay $
% -------------------------------------------------------------------------
function [lme,tmp] = bicp_evidence_bb(ks,ns,nSamples,burnIn)
    
    % Initialization
    m = size(ns,2);   % number of subjects
    assert(size(ks,1)==1, 'ks must be a 1 x SUBJECT row vector');
    assert(all(ks<=ns));
    try, nSamples; catch; nSamples = 10000; end
    try, burnIn; catch; burnIn = []; end
    
    % Collect samples from p(alpha,beta|m)
    [alphas,betas] = bicp_prior_bb(nSamples,burnIn);
    
    nSamples = length(alphas);
    
    % Compute model evidence
    % (Here we're using the variant with alpha/beta rather than pi as
    % describred in the manuscript.)
    acc = NaN(nSamples,1);
    parfor t=1:nSamples
        pis = betarnd(alphas(t),betas(t),1,m);
        tmp = exp(nansum(log(binopdf(ks,ns,pis))));
        acc(t) = tmp;
    end
    lme = -log(nSamples) + log(nansum(acc));
    
    % Check return values
    if isnan(lme), warning('model evidence turned out NaN'); end
    
    tmp = acc;
end

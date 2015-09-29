% Computes an approximation to the log model evidence (marginal likelihood)
% of the twofold beta-binomial model.
%
% Usage:
%     lme = bicp_evidence_tbb(ks,ns,nSamples)
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
% $Id: bicp_evidence_tbb.m 16174 2012-05-29 12:34:01Z bkay $
% -------------------------------------------------------------------------
function [lme,tmp] = bicp_evidence_tbb(ks,ns,nSamples)
    
    % Initialization
    m = size(ns,2);
    assert(size(ks,1)==2, 'ks must be a 2 x SUBJECT row vector');
    assert(all(all(ks<=ns)));
    try, nSamples; catch; nSamples = 1e5; end
    assert(logical(isint(nSamples)),'nSamples must be an integer');
    assert(nSamples>0,'nSamples must be greater than 0');
    
    % Compute log model evidence
    lme = bicp_evidence_bb(ks(1,:),ns(1,:),nSamples) ...
        + bicp_evidence_bb(ks(2,:),ns(2,:));
end

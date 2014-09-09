% Compares two alternative models for mixed-effects inference on the
% balanced accuracy:
% 
%     (1) the twofold beta-binomial model (tbb)
%     (2) the bivariate normal-binomial model (bnb)
% 
% Usage:
%     logBF = bicp_bms(ks,ns)
%     logBF = bicp_bms(ks,ns,nSamples)
%     [logBF,lmes] = bicp_bms(ks,ns,nSamples)
%
% Arguments:
%     ks: 2 x SUBJECTS matrix
%     ns: 2 x SUBJECTS matrix
%     nSamples: number of samples for model inversion (default=1e5)
%
% Return values:
%     logBF: The log Bayes factor (BF) summarizes the relative evidence for
%         the two models. For its interpretation, see:
%         http://en.wikipedia.org/wiki/Bayes_factor
%     lmes: vector of individual log model evidences

% Kay H. Brodersen, ETH Zurich
% $Id: bicp_bms.m 16210 2012-05-31 07:04:48Z bkay $
% -------------------------------------------------------------------------
function [logBF,lmes] = bicp_bms(ks,ns,nSamples,verbose)
    
    % Check input
    try, nSamples; catch; nSamples = 1e5; end
    try, verbose; catch; verbose = 1; end
    [ks,ns] = check_ks_ns(ks,ns);
    assert(size(ks,1)==2 && size(ns,1)==2,'ks and ns must each contain two rows');
    
    % Log model evidence of the 'tbb' model
    lmes(1) = bicp_evidence_bb(ks(1,:),ns(1,:),nSamples) + bicp_evidence_bb(ks(2,:),ns(2,:),nSamples);
    
    % Log model evidence of the 'bnb' model
    lmes(2) = bicp_evidence_bnb(ks,ns,nSamples);
    
    % Compute log Bayes Factor
    logBF = lmes(1) - lmes(2);
    
    % Print result
    if verbose>0
        disp('Bayesian model comparison:');
        disp(['log BF = ',num2str(logBF)]);
        if (logBF>0)
            disp(['There is ''',bf2str(exp(logBF)),''' evidence in favour of the twofold beta-binomial (tbb) model.']);
        elseif (logBF<0)
            disp(['There is ''',bf2str(exp(-logBF)),''' evidence in favour of the bivariate normal-binomial (bnb) model.']);
        else
            disp(['BF is zero']);
        end
    end
    
end

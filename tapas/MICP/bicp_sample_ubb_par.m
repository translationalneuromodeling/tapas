% Metropolis algorithm to sample from the posterior p(alpha,beta|ks),
% from the posteriors p(pi_j|ks), and from the posterior predictive
% densities p(pi~|ks). Based on the beta-binomial model.
%
% Parallelized version.
%
% See also:
%     bicp_sample_ubb

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_sample_ubb_par.m 16211 2012-05-31 07:05:25Z bkay $
% -------------------------------------------------------------------------
function [alphas,betas,pijs,pis] = bicp_sample_ubb_par(ks,ns,nSamples,varargin)
    
    % Hard-wired: number of cores to use
    B = 8;
    func = @bicp_sample_ubb;
    
    % Initialization
    assert(mod(nSamples,B)==0,['nSamples must be a multiple of ',num2str(B)]);
    nSamples_each = ceil(nSamples/B);
    m = size(ks,2);
    
    % Step 1: compute parallel chains
    chains = struct;
    tmp_nargout = nargout;
    parfor b=1:B
        tmp_alphas = [];
        tmp_betas = [];
        tmp_pijs = [];
        tmp_pis = [];
        if tmp_nargout==2
            [tmp_alphas,tmp_betas] = ...
                func(ks,ns,nSamples_each,varargin{:});
        elseif tmp_nargout==3
            [tmp_alphas,tmp_betas,tmp_pijs] = ...
                func(ks,ns,nSamples_each,varargin{:});
        else
            [tmp_alphas,tmp_betas,tmp_pijs,tmp_pis] = ...
                func(ks,ns,nSamples_each,varargin{:});
        end
        chains(b).alphas = tmp_alphas;
        chains(b).betas = tmp_betas;
        chains(b).pijs = tmp_pijs;
        chains(b).pis = tmp_pis;
    end
    
    % Todo: evaluate stopping rule
    % if ~bicp_eval_chains(chains), ...
    
    % Step 2: concatenate individual chains
    alphas = NaN(1,nSamples);
    betas = NaN(1,nSamples);
    pijs = NaN(m,nSamples);
    pis = NaN(1,nSamples);
    for b=1:B
        from = (b-1)*nSamples_each+1;
        to = b*nSamples_each;
        alphas(:,from:to) = chains(b).alphas;
        betas(:,from:to) = chains(b).betas;
        if tmp_nargout>=3
            pijs(:,from:to) = chains(b).pijs;
        end
        if tmp_nargout>=4
            pis(:,from:to) = chains(b).pis;
        end
    end
end

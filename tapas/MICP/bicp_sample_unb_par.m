% Gibbs sampling algorithm to sample from the posterior p(mu,lambda|ks),
% from the posteriors p(pi_j|ks), and from the posterior predictive density
% p(pi|ks,ns). Based on the univariate normal-binomial model.
% 
% Parallelized version.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_sample_unb_par.m 13703 2012-01-07 20:06:16Z bkay $
% -------------------------------------------------------------------------
function [mus,lambdas,pijs,pis] = bicp_sample_unb_par(ks,ns,nSamples,varargin)
    
    % Hard-wired: number of cores
    B = 8;
    
    % Initialization
    assert(mod(nSamples,B)==0,['nSamples must be a multiple of ',num2str(B)]);
    nSamples_each = ceil(nSamples/B);
    if length(varargin)>=2
        assert(varargin{2}==0, 'verbose must be 0 in parallel mode');
    end
    m = size(ks,2);
    
    % Step 1: compute parallel chunks
    loops = struct;
    tmp_nargout = nargout;
    parfor j=1:B
        tmp_mus=[]; tmp_lambdas=[]; tmp_pijs=[]; tmp_pis=[];
        switch(tmp_nargout)
            case 1
            [tmp_mus] = ...
                bicp_sample_unb(ks,ns,nSamples_each,varargin{:});
            case 2
            [tmp_mus,tmp_lambdas] = ...
                bicp_sample_unb(ks,ns,nSamples_each,varargin{:});
            case 3
            [tmp_mus,tmp_lambdas,tmp_pijs] = ...
                bicp_sample_unb(ks,ns,nSamples_each,varargin{:});
            case 4
            [tmp_mus,tmp_lambdas,tmp_pijs,tmp_pis] = ...
                bicp_sample_unb(ks,ns,nSamples_each,varargin{:});
        end
        loops(j).mus = tmp_mus;
        loops(j).lambdas = tmp_lambdas;
        loops(j).pijs = tmp_pijs;
        loops(j).pis = tmp_pis;
    end
    
    % Step 2: concatenate individual contributions
    mus = NaN(1,nSamples);
    lambdas = NaN(1,nSamples);
    pijs = NaN(m,nSamples);
    pis = NaN(1,nSamples);
    for b=1:B
        from = (b-1)*nSamples_each+1;
        to = b*nSamples_each;
        mus(:,from:to) = loops(b).mus;
        if tmp_nargout>=2
            lambdas(:,from:to) = loops(b).lambdas;
        end
        if tmp_nargout>=3
            pijs(:,from:to) = loops(b).pijs;
        end
        if tmp_nargout>=4
            pis(1,from:to) = loops(b).pis;
        end
    end
end

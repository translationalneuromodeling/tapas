% Gibbs sampling algorithm to sample from the posterior p(mu,Sigma|ks),
% from the posteriors p(pi_j|ks), and from the posterior predictive density
% p(pi|ks,ns). Based on the normal-binomial model.
% 
% Parallelized version.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: bicp_sample_bnb_par.m 15622 2012-04-28 11:23:58Z bkay $
% -------------------------------------------------------------------------
function [mus,Sigmas,pijs,pis] = bicp_sample_bnb_par(ks,ns,nSamples,varargin)
    
    % Hard-wired: number of cores
    B = 8;
    
    % Initialization
    assert(mod(nSamples,B)==0,['nSamples must be a multiple of ',num2str(B)]);
    nSamples_each = ceil(nSamples/B);
    m = size(ks,2);
    
    % Step 1: compute parallel chunks
    loops = struct;
    tmp_nargout = nargout;
    parfor j=1:B
        tmp_mus=[]; tmp_Sigmas=[]; tmp_pijs=[]; tmp_pis=[];
        switch(tmp_nargout)
            case 1
            [tmp_mus] = ...
                bicp_sample_bnb(ks,ns,nSamples_each,varargin{:});
            case 2
            [tmp_mus,tmp_Sigmas] = ...
                bicp_sample_bnb(ks,ns,nSamples_each,varargin{:});
            case 3
            [tmp_mus,tmp_Sigmas,tmp_pijs] = ...
                bicp_sample_bnb(ks,ns,nSamples_each,varargin{:});
            case 4
            [tmp_mus,tmp_Sigmas,tmp_pijs,tmp_pis] = ...
                bicp_sample_bnb(ks,ns,nSamples_each,varargin{:});
        end
        loops(j).mus = tmp_mus;
        loops(j).Sigmas = tmp_Sigmas;
        loops(j).pijs = tmp_pijs;
        loops(j).pis = tmp_pis;
    end
    
    % Step 2: concatenate individual contributions
    mus = NaN(2,nSamples);
    Sigmas = NaN(2,2,nSamples);
    pijs = NaN(2,m,nSamples);
    pis = NaN(2,nSamples);
    for b=1:B
        from = (b-1)*nSamples_each+1;
        to = b*nSamples_each;
        mus(:,from:to) = loops(b).mus;
        if tmp_nargout>=2
            Sigmas(:,:,from:to) = loops(b).Sigmas;
        end
        if tmp_nargout>=3
            pijs(:,:,from:to) = loops(b).pijs;
        end
        if tmp_nargout>=4
            pis(:,from:to) = loops(b).pis;
        end
    end
end

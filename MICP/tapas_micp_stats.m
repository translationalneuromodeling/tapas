% Mixed-effects inference on classification performance.
%
% Usage:
%     mu = tapas_micp_stats(ks,ns)
%     [mu,p,ci] = tapas_micp_stats(ks,ns,...)
%     [mu,p,ci,stats] = tapas_micp_stats(ks,ns,...)
%
% Arguments:
%     When inferring on accuracies, the input arguments are:
%     ks:    1xm vector of number of correctly predicted trials (for m subjects)
%     ns:    1xm vector of total number of trials (for m subjects)
%     
%     When inferring on balanced accuracies, the input arguments are:
%     ks:    2xm matrix of correctly predicted positive trials (first row)
%            and corectly predicted negative trials (second row)
%     ns:    2xm matrix of total number of positive (first row) and negative
%            (second row) trials
% 
% Optional named arguments:
%     'model': specifies model assumptions and inference algorithm
% 
%            Available models for inference on accuracies:
%            'unb_vb': univariate normal-binomial model using VB (default)
%            'unb_mcmc': univariate normal-binomial model using MCMC
%            'ubb_mcmc': univariate beta-binomial model using MCMC
%            
%            Available models for inference on balanced accuracies:
%            'tnb_vb': twofold normal-binomial model using VB (default)
%            'tbb_mcmc': twofold beta-binomial model using MCMC
%            'bnb_mcmc': bivariate normal-binomial model using MCMC
%         
%     'nSamples': specifies the number of MCMC samples (default: 1e4).
%            More samples take more time but yield more accurate results.
% 
% Return values:
%     mu:    Posterior mean of the population mean accuracy or balanced
%            accuracy. This is the expected performance of the classifier
%            at the group level.
%     p:     Posterior infraliminal probability of the population mean.
%            This is the posterior belief that the classifier did not
%            operate above chance (50%). A small infraliminal probability
%            represents evidence for above-chance performance.
%     ci:    Posterior 95% credible interval of the population mean. This
%            interval can be used to show error bars around mu.
%     stats: Additional return values, depending on the selected model. See
%            individual inference functions for details.
%
% This function provides an interface to a set of underlying inference
% algorithms. For details, see their respective help texts.
% 
% Example:
%     ks = [19 41 15 39 39; 41 46 43 48 37];
%     ns = [45 51 20 46 58; 55 49 80 54 42];
%     [mu,p,ci] = tapas_micp_stats(ks,ns);
% 
% Literature:
%     K.H. Brodersen, C. Mathys, J.R. Chumbley, J. Daunizeau, C.S. Ong,
%     J.M. Buhmann, & K.E. Stephan. Mixed-effects inference on
%     classification performance in hierarchical datsets (in revision).
%     
%     K.H. Brodersen, C.S. Ong, J.M. Buhmann, & K.E. Stephan. The balanced
%     accuracy and its posterior distribution. ICPR (2010), 3121-3124.
% 
% Kay H. Brodersen, ETH Zurich, khbrodersen@gmail.com

% $Id: tapas_micp_stats.m 19997 2013-10-05 21:02:59Z bkay $
% -------------------------------------------------------------------------
function [mu,p,ci,stats] = tapas_micp_stats(ks,ns,varargin)
    
    % Check input
    [ks,ns] = check_ks_ns(ks,ns);
    
    % Process additional arguments
    if size(ks,1)==1, defaults.model = 'unb_vb'; end
    if size(ks,1)==2, defaults.model = 'tnb_vb'; end
    defaults.nSamples = 1e4;
    args = propval(varargin,defaults);
    
    % Inference
    switch lower(args.model)
    
    % Univariate normal-binomial, VB
    case 'unb_vb'
        assert(size(ks,1)==1 && size(ns,1)==1,'for inference on accuracies, ks and ns must be row vectors');
        q = vbicp_unb(ks,ns);
        mu = logitnmean(q.mu_mu,1/sqrt(q.eta_mu));
        if nargout >= 2, p = logitncdf(0.5,q.mu_mu,1/sqrt(q.eta_mu)); end
        if nargout >= 3, ci = [logitninv(0.025,q.mu_mu,1/sqrt(q.eta_mu)), logitninv(0.975,q.mu_mu,1/sqrt(q.eta_mu))]; end
        stats.q = q;
    
    % Univariate normal-binomial, MCMC
    case 'unb_mcmc'
        assert(size(ks,1)==1 && size(ns,1)==1,'for inference on accuracies, ks and ns must be row vectors');
        [mus,lambdas] = bicp_sample_unb(ks,ns,args.nSamples);
        stats.mus = sigm(mus);
        stats.lambdas = lambdas;
        mu = mean(stats.mus);
        if nargout >= 2, p  = sum(stats.mus<0.5)/length(stats.mus); end
        if nargout >= 3, ci = [percentile(stats.mus,2.5), percentile(stats.mus,97.5)]; end
    
    % Univariate beta-binomial, MCMC
    case 'ubb_mcmc'
        assert(size(ks,1)==1 && size(ns,1)==1,'for inference on accuracies, ks and ns must be row vectors');
        [alphas,betas] = bicp_sample_ubb(ks,ns,args.nSamples);
        stats.mus = alphas./(alphas+betas);
        stats.alphas = alphas;
        stats.betas = betas;
        mu = mean(stats.mus);
        if nargout >= 2, p  = sum(stats.mus<0.5)/length(stats.mus); end
        if nargout >= 3, ci = [percentile(stats.mus,2.5), percentile(stats.mus,97.5)]; end
    
    % Twofold normal-binomial, VB
    case 'tnb_vb'
        assert(size(ks,1)==2 && size(ns,1)==2,'for inference on balanced accuracies, ks and ns must each contain two rows');
        qp = vbicp_unb(ks(1,:),ns(1,:));
        qn = vbicp_unb(ks(2,:),ns(2,:));
        mu = logitnavgmean(qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu));
        if nargout >=2, p = logitnavgcdf(0.5,qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu)); end
        if nargout >=3
            ci = [logitnavginv(0.025,qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu)), ...
                  logitnavginv(0.975,qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu))];
        end
        stats.qp = qp;
        stats.qn = qn;
        for j=1:size(ks,2)
            stats.mu_phij(j) = logitnavgmean(qp.mu_rho(j),1/sqrt(qp.eta_rho(j)),qn.mu_rho(j),1/sqrt(qn.eta_rho(j)));
        end
    
    % Twofold beta-binomial, MCMC
    case 'tbb_mcmc'
        assert(size(ks,1)==2 && size(ns,1)==2,'for inference on balanced accuracies, ks and ns must each contain two rows');
        [alphas_p,betas_p] = bicp_sample_ubb(ks(1,:),ns(1,:),round(args.nSamples));
        [alphas_n,betas_n] = bicp_sample_ubb(ks(2,:),ns(2,:),round(args.nSamples));
        stats.mus = 0.5*(alphas_p./(alphas_p+betas_p) + alphas_n./(alphas_n+betas_n));
        stats.alphas_p = alphas_p; stats.betas_p = betas_p;
        stats.alphas_n = alphas_n; stats.betas_n = betas_n;
        mu = mean(stats.mus);
        if nargout >= 2, p  = sum(stats.mus<0.5)/length(stats.mus); end
        if nargout >= 3, ci = [percentile(stats.mus,2.5), percentile(stats.mus,97.5)]; end
        
    % Bivariate normal-binomial, MCMC
    case 'bnb_mcmc'
        assert(size(ks,1)==2 && size(ns,1)==2,'for inference on balanced accuracies, ks and ns must each contain two rows');
        mus = bicp_sample_bnb(ks,ns,args.nSamples);
        stats.mus = mean(sigm(mus),2);
        mu = mean(stats.mus,1);
        if nargout >= 2, p  = sum(stats.mus<0.5)/length(stats.mus); end
        if nargout >= 3, ci = [percentile(stats.mus,2.5), percentile(stats.mus,97.5)]; end
        
    % Invalid model
    otherwise
        error('invalid model - type ''help tapas_micp_stats'' for help');
    end
    
end

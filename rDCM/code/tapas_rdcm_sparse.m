function [ output ] = tapas_rdcm_sparse(DCM, X, Y, args)
% [ output ] = tapas_rdcm_sparse(DCM, X, Y, args)
% 
% Variational Bayesian inversion of a linear DCM with regression DCM, 
% including sparsity constrains on the connectivity parameters. The
% function implements the VB update equations derived in Frässle et al.
% (2018).
% 
%   Input:
%   	DCM             - model structure
%       X               - design matrix (predictors)
%       Y               - data
%       args            - arguments
%
%   Output:
%       output          - output structure
% 
%   Reference:
%       Frässle, S., Lomakina, E.I., Kasper, L., Manjaly Z.M., Leff, A., 
%       Pruessmann, K.P., Buhmann, J.M., Stephan, K.E., 2018. A generative 
%       model of whole-brain effective connectivity. NeuroImage 179, 505-529.
%       https://doi.org/10.1016/j.neuroimage.2018.05.058
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2021 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% number of regions
nr = size(DCM.Y.y,2);

% precision limit
pr = 10^(-5);

% add confound regressor dimensions
Nc = size(DCM.U.X0,2);
for nc = 1:Nc
    DCM.b(:,:,end+1) = DCM.b(:,:,1);
    DCM.c(:,end+1)   = ones(1,size(DCM.c,1));
end

% no confound regressors for simulations
if ( strcmp(args.type,'s') )
    DCM.c(:,end-Nc+1:end) = 0;
end

% get the priors
[m0, l0, a0, b0] = tapas_rdcm_get_prior_all(DCM);

% effective dimensionality
idx_x = [ones(1,size(DCM.a,1)), zeros(1,size(DCM.b,1)*size(DCM.b,3)), ones(1,size(DCM.c,2))] == 1;
D     = sum(idx_x);
    
% define a random order for the regions
ord_regions = 1:nr;

% results arrays (per iteration)
mN   = zeros(nr,D);
sN   = cell(nr,1);
aN   = zeros(nr,1);
bN   = zeros(nr,1);
z    = zeros(nr,D);
logF = zeros(1,nr);

% define signal array
yd_pred_rdcm_fft = NaN(size(DCM.Y.y));


% default settings for model inversion
if ( ~isfield(args,'iter') ),             args.iter = 100; end
if ( ~isfield(args,'restrictInputs') ),   args.restrictInputs = 1; end
if ( ~isfield(args,'diagnostics_logF') ), args.diagnostics_logF = 0; end


% check if many p0s are estimated - disable output of individual regions
if ( args.verbose == 0 )
    DCM.M.noprint = 1;
end


% if progress for regions is outputed
if ( ~isfield(DCM,'M') || ~isfield(DCM.M,'noprint') || ~DCM.M.noprint )
    fprintf('\n')
    reverseStr = '';
end

% iterate over regions
for k = ord_regions
    
    % output progress of regions
    if ( ~isfield(DCM,'M') || ~isfield(DCM.M,'noprint') || ~DCM.M.noprint )
        msg = sprintf('Processing region: %d/%d', k, length(ord_regions));
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    
    % results arrays (per region)
    mN_iter             = cell(args.iter,1);
    sN_iter             = cell(args.iter,1);
    aN_iter             = cell(args.iter,1);
    bN_iter             = cell(args.iter,1);
    z_iter              = cell(args.iter,1);
    l0_iter             = cell(args.iter,1);
    logF_iter           = NaN(args.iter,1);
    log_lik_iter        = NaN(args.iter,1);
    log_p_weight_iter	= NaN(args.iter,1);
    log_p_prec_iter     = NaN(args.iter,1);
    log_p_z_iter        = NaN(args.iter,1);
    log_q_weight_iter	= NaN(args.iter,1);
    log_q_prec_iter     = NaN(args.iter,1);
    log_q_z_iter        = NaN(args.iter,1);
    
    % iterate over the number of re-initializations
    for iter = 1:args.iter

        % filter unnecessary data
        idx_y = ~isnan(Y(:,k));
        X_r   = X(idx_y,:);
        Y_r   = Y(idx_y,k);

        % effective number of data points
        N_eff = sum(idx_y);
        
        % get the priors per regions
        l0_r = diag(l0(k,idx_x));
        m0_r = m0(k,idx_x)';

        % set p0 (feature informativeness)
        try
            p0 = ones(D,1)*args.p0_temp;
        catch
            p0 = ones(D,1)*0.5;
        end
        
        % inform p0 (e.g., by anatomical information) / information needs to be expressed in DCM.a
        if ( isfield(args,'p0_inform') && args.p0_inform )
            p0(1:size(DCM.a,1)) = p0(1:size(DCM.a,1)).*DCM.a(k,:)';
        end
        
        % ensure self-connectivity
        p0(k) = 1;
        
        % ensure baseline regressor (will only contribute for empirical data)                                                                                                                                                                                                                                                 
        if ( strcmp(args.type,'s') )
            p0(end) = 0;
        else
            p0(end) = 1;
        end
        
        % make sure that the driving inputs are only on the correct connections
        if ( args.restrictInputs == 1 )
            p0(end-(size(DCM.c,2)-1):end-1) = DCM.c(k,1:end-1);
        end
        
        % results array per regions
        sN_r = zeros(size(l0_r));
        mN_r = zeros(size(m0_r));
        
        
        %% initalize
        
        % estimate variables X'X and X'Y per region
        W = X_r(:,idx_x)'*X_r(:,idx_x);
        v = X_r(:,idx_x)'*Y_r;
        
        % intialize z, t & aN per regions
        z_r   = p0;
        t     = a0/b0;
        aN_r  = a0 + N_eff/(2*args.r_dt);
        
        
        % cycle stops after 500 iterations;
        count = 500;
        
        
        %% updating mN, sN

        % define random matrix Z
        Z = diag(z_r);
        
        % expectation (over Z) of ZX'XZ
        G = Z*W*Z; 
        G(eye(D)>0) = z_r.*diag(W);

        % set old F
        logF_old = -inf;
        
        % convergence criterion
        convergence = 0;

        % loop until convergence
        while ~convergence
            
            
            % update covariance and mean of parameters
            sN_r = inv(t*G + l0_r);
            mN_r = sN_r*(t*Z*v + l0_r*m0_r);


            %% updating z

            % estimate some factors
            Wb = W.*(mN_r*mN_r');
            Ws = W.*sN_r;
            A  = Wb + Ws;

            % estimate g
            g = log(p0./(1-p0)) + t*mN_r.*v + t*diag(A)/2;
            
            % define a random order
            ord = randperm(D);
            
            % iterate through all variables
            for i = ord
                z_r(i) = 1;
                g(i) = g(i) - t*z_r'*A(:,i);
                z_r(i) = 1/(1+exp(-g(i)));
            end


            %% updating mN, sN

            % build random matrix Z
            Z = diag(z_r);

            % re-estimate expectation (over Z) of ZX'XZ
            G = Z*W*Z; 
            G(eye(D)>0) = z_r.*diag(W);

            %% updating bN

            % update bN per region
            QF = Y_r'*Y_r/2 - mN_r'*Z*v + mN_r'*G*mN_r/2 + trace(G*sN_r)/2;
            bN_r = b0 + QF;

            % update tau
            t = aN_r/bN_r;
            
            
            % check for sparsity (because of small values)
            mN_r(abs(mN_r)<10^(-5)) = 0;
            z_r(mN_r==0)            = 0;

            % get the "present" connections
            z_idx  = (z_r>pr^2).*(z_r<1)>0;


            %% compute model evidence

            % compute the components of the model evidence
            log_lik = N_eff*(psi(aN_r) - log(bN_r))/2 - N_eff*log(2*pi)/2 - t*QF/2;
            log_p_weight = 1/2*tapas_rdcm_spm_logdet(l0_r) - D*log(2*pi)/2 - (mN_r-m0_r)'*l0_r*(mN_r-m0_r)/2 - trace(l0_r*sN_r)/2;
            log_p_prec = a0*log(b0) - gammaln(a0) + (a0-1)*(psi(aN_r) - log(bN_r)) - b0*t;
            log_p_z = sum(log(1-p0(z_idx)) + z_r(z_idx).*log(p0(z_idx)./(1-p0(z_idx))));
            log_q_weight = 1/2*tapas_rdcm_spm_logdet(sN_r) + D*(1+log(2*pi))/2;
            log_q_prec = aN_r - log(bN_r) + gammaln(aN_r) + (1-aN_r)*psi(aN_r);
            log_q_z = sum(-(1-z_r(z_idx)).*log(1-z_r(z_idx)) - z_r(z_idx).*log(z_r(z_idx)));

            % compute the negative free energy per region
            logF_temp = real(log_lik + log_p_prec + log_p_weight + log_p_z + log_q_prec + log_q_weight + log_q_z);
            
            % check whether convergence is reached
            if ( ((logF_old - logF_temp)^2 < pr^2) )
                convergence = 1;
            end
            
            % store old negative free energy
            logF_old = logF_temp;

            % decrese cycle counter and check for end
            count = count - 1;
            
            % end optimization when number of iterations is reached
            if count<0
                break;
            end
        end


        % check for sparsity (because of small values)
        mN_r(abs(mN_r)<10^(-5)) = 0;
        z_r(mN_r==0)            = 0;
        
        % get the "present" connections
        z_idx  = (z_r>pr^2).*(z_r<1)>0;


        %% compute model evidence

        % compute the components of the model evidence
        log_lik_iter(iter)      = N_eff*(psi(aN_r) - log(bN_r))/2 - N_eff*log(2*pi)/2 - t*QF/2;
        log_p_weight_iter(iter) = 1/2*tapas_rdcm_spm_logdet(l0_r) - D*log(2*pi)/2 - (mN_r-m0_r)'*l0_r*(mN_r-m0_r)/2 - trace(l0_r*sN_r)/2;
        log_p_prec_iter(iter)   = a0*log(b0) - gammaln(a0) + (a0-1)*(psi(aN_r) - log(bN_r)) - b0*t;
        log_p_z_iter(iter)      = sum(log(1-p0(z_idx)) + z_r(z_idx).*log(p0(z_idx)./(1-p0(z_idx))));
        log_q_weight_iter(iter) = 1/2*tapas_rdcm_spm_logdet(sN_r) + D*(1+log(2*pi))/2;
        log_q_prec_iter(iter)   = aN_r - log(bN_r) + gammaln(aN_r) + (1-aN_r)*psi(aN_r);
        log_q_z_iter(iter)      = sum(-(1-z_r(z_idx)).*log(1-z_r(z_idx)) - z_r(z_idx).*log(z_r(z_idx)));

        % compute the negative free energy per region
        logF_iter(iter) = real(log_lik_iter(iter) + log_p_prec_iter(iter) + log_p_weight_iter(iter) + log_p_z_iter(iter) + log_q_prec_iter(iter) + log_q_weight_iter(iter) + log_q_z_iter(iter));
        
        % asign the iteration-specific values
        mN_iter{iter} = mN_r;
        z_iter{iter}  = z_r;
        l0_iter{iter} = l0_r;
        sN_iter{iter} = sN_r;
        aN_iter{iter} = aN_r;
        bN_iter{iter} = bN_r;
        
        % clear the variables
        clear mN_r z_r l0_r sN_r aN_r bN_r
        
    end
    
    % get the logF for all iterations
    [~, best] = max(logF_iter);
    
    % store region-specific parameters
    mN(k,:)             = real(mN_iter{best});
    z(k,:)              = real(z_iter{best});
    sN{k}(idx_x,idx_x)	= real(sN_iter{best});
    aN(k)               = real(aN_iter{best});
    bN(k)               = real(bN_iter{best});
    logF(k)             = logF_iter(best);
    
    % get the predicted signal from the GLM (in frequency domain)
    yd_pred_rdcm_fft(:,k)	= X(:,idx_x) * mN_iter{best};
    
    % store the region-specific components of logF (for diagnostics)
    if ( args.diagnostics_logF )
        logF_term.log_lik(k)        = log_lik_iter(best);
        logF_term.log_p_weight(k)   = log_p_weight_iter(best);
        logF_term.log_p_prec(k)     = log_p_prec_iter(best);
        logF_term.log_p_z(k)        = log_p_z_iter(best);
        logF_term.log_q_weight(k)   = log_q_weight_iter(best);
        logF_term.log_q_prec(k)     = log_q_prec_iter(best);
        logF_term.log_q_z(k)        = log_q_z_iter(best);
    else
        logF_term = [];
    end
    
    
    % clear the region-specific parameters
    clear mN_iter z_iter l0_iter sN_iter aN_iter bN_iter logF_iter
    clear log_lik_iter log_p_weight_iter log_p_prec_iter log_p_z_iter log_q_weight_iter log_q_prec_iter log_q_z_iter
    
end

% write the results to the output file
output{1} = tapas_rdcm_store_parameters(DCM, mN, sN, aN, bN, logF, logF_term, idx_x, z, args);

% store the priors
output{1}.priors.m0 = m0;
output{1}.priors.l0 = l0;
output{1}.priors.a0 = a0;
output{1}.priors.b0 = b0;
output{1}.priors.p0 = args.p0_temp;

% store the rDCM variant
output{1}.inversion = 'tapas_rdcm_sparse';

% store the true and predicted temporal derivatives (in frequency domain)
output{1}.temp.yd_source_fft    = Y;
output{1}.temp.yd_pred_rdcm_fft = yd_pred_rdcm_fft;

end

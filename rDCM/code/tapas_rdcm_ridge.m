function [ output ] = tapas_rdcm_ridge(DCM, X, Y, args)
% [ output ] = tapas_rdcm_ridge(DCM, X, Y, args)
% 
% Variational Bayesian inversion of a linear DCM with regression DCM. The
% function implements the VB update equations derived in Frässle et al.
% (2017).
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
%       Frässle, S., Lomakina, E.I., Razi, A., Friston, K.J., Buhmann, J.M., 
%       Stephan, K.E., 2017. Regression DCM for fMRI. NeuroImage 155, 406-421.
%       https://doi.org/10.1016/j.neuroimage.2017.02.090
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


% precision limit
pr = 10^(-5);

% add confound regressor dimensions
Nc = size(DCM.U.X0,2);
for nc = 1:Nc
    DCM.b(:,:,end+1) = DCM.b(:,:,1);
    DCM.c(:,end+1)   = ones(1,size(DCM.c,1));
end

% get the number of regions and inputs
[nr, nu] = size(DCM.c);

% no baseline regressor for simulations
if ( strcmp(args.type,'s') )
    DCM.c(:,end) = 0;
end

% define the relevant parameters and get priors
idx = [DCM.a reshape(DCM.b,nr,nr*nu) DCM.c]>0;
[m0, l0, a0, b0] = tapas_rdcm_get_prior(DCM);


% define the results arrays
mN   = zeros(size(idx));
sN   = cell(nr,1);
aN   = zeros(nr,1);
bN   = zeros(nr,1);
logF = zeros(1,nr);

% define array for predicted derivative of signal (in frequency domain)
yd_pred_rdcm_fft = NaN(size(DCM.Y.y));


% iterate over regions
for k = 1:nr
    
    % find finite values
    idx_y = ~isnan(Y(:,k));
    
    % prepare regressors and data by removing unnecessary dimensions
    X_r = X(idx_y,idx(k,:)');
    Y_r = Y(idx_y,k);
    
    % effective number of data points
    N_eff = sum(idx_y)/(args.r_dt);
    
    % effective dimensionality
    D_r = sum(idx(k,:));
    
    
    %% read priors
    
    % prior precision matrix
    l0_r = diag(l0(k,idx(k,:)));
    
    % prior means
    m0_r = m0(k,idx(k,:))';
    
    % compute X'X and X'Y
    W = X_r'*X_r;
    v = X_r'*Y_r;
    
    
    %% compute optimal theta and tau
    
    % initial value for tau
    t = a0/b0;
    
    % estimate alpha_N
    aN_r = a0 + N_eff/ (2*args.r_dt);
    
    % cycle stops after 500 iterations
    count = 500; 
    
    % set old F
    logF_old = -inf;
    
    % convergence criterion
    convergence = 0;
    
    % loop until convergence
    while ~convergence
        
        % update posterior covariance matrix
        sN_r = inv(t*W + l0_r);
        
        % update posterior means
        mN_r = sN_r*(t*v + l0_r*m0_r);
        
        % update posterior rate parameter
        QF = (Y_r-X_r*mN_r)'*(Y_r-X_r*mN_r)/2 + trace(W*sN_r)/2;
        bN_r = b0 + QF;
        
        % update tau
        t = aN_r/bN_r;
        
        
        %% compute model evidence
    
        % compute components of the model evidence
        log_lik      = N_eff*(psi(aN_r) - log(bN_r))/2 - N_eff*log(2*pi)/2 - QF*t;
        log_p_weight = 1/2*tapas_rdcm_spm_logdet(l0_r) - D_r*log(2*pi)/2 - (mN_r-m0_r)'*l0_r*(mN_r-m0_r)/2 - trace(l0_r*sN_r)/2;
        log_p_prec   = a0*log(b0) - gammaln(a0) + (a0-1)*(psi(aN_r) - log(bN_r)) - b0*t;
        log_q_weight = 1/2*tapas_rdcm_spm_logdet(sN_r) + D_r*(1+log(2*pi))/2;
        log_q_prec   = aN_r - log(bN_r) + gammaln(aN_r) + (1-aN_r)*psi(aN_r);

        % compute the negative free energy per region
        logF(k) = log_lik + log_p_prec + log_p_weight + log_q_prec + log_q_weight;
        
        % check whether convergence is reached
        if ( ((logF_old - real(logF(k)))^2 < pr^2) )
            convergence = 1;
        end

        % store old negative free energy
        logF_old = real(logF(k));
        
        % decease the counter
        count = count - 1;
        
        % end optimization when number of iterations is reached
        if count<0
            break;
        end
    end
    
    
    %% re-compute model evidence
    
    % expected log likelihood
    log_lik = N_eff*(psi(aN_r) - log(bN_r))/2 - N_eff*log(2*pi)/2 - QF*t;
    
    % expected ln p(theta)
    log_p_weight = 1/2*tapas_rdcm_spm_logdet(l0_r) - D_r*log(2*pi)/2 - (mN_r-m0_r)'*l0_r*(mN_r-m0_r)/2 - trace(l0_r*sN_r)/2;
    
    % expected ln p(tau)
    log_p_prec = a0*log(b0) - gammaln(a0) + (a0-1)*(psi(aN_r) - log(bN_r)) - b0*t;
    
    % expected ln q(theta)
    log_q_weight = 1/2*tapas_rdcm_spm_logdet(sN_r) + D_r*(1+log(2*pi))/2;
    
    % expected ln q(tau)
    log_q_prec = aN_r - log(bN_r) + gammaln(aN_r) + (1-aN_r)*psi(aN_r);
    
    % region-specific negative free energy
    logF(k) = log_lik + log_p_prec + log_p_weight + log_q_prec + log_q_weight;
    
    % store region-specific parameters
    mN(k,idx(k,:))           = real(mN_r);
    sN{k}(idx(k,:),idx(k,:)) = real(sN_r);
    aN(k)                    = real(aN_r);
    bN(k)                    = real(bN_r);
    
    % get the predicted signal from the GLM (in frequency domain)
    yd_pred_rdcm_fft(:,k)	= X(:,idx(k,:)') * mN_r;
    
end

% store parameters
output = tapas_rdcm_store_parameters(DCM, mN, sN, aN, bN, real(logF), [], idx, [], args);

% store the priors
output.priors.m0 = m0;
output.priors.l0 = l0;
output.priors.a0 = a0;
output.priors.b0 = b0;

% store the rDCM variant
output.inversion = 'tapas_rdcm_ridge';

% store the true and predicted temporal derivatives (in frequency domain)
output.temp.yd_source_fft    = Y;
output.temp.yd_pred_rdcm_fft = yd_pred_rdcm_fft;

end

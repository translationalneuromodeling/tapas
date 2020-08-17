function [ obj ] = vb_invert( obj )
% Run VB update equations until convergence.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 



%% initialization
obj = obj.vb_init( );
bUpdateKmeans       = false;
bUpdateAssignments  = false;
bUpdateClusters     = false;
bVerbose            = obj.options.nvp.verbose;
nIt = obj.options.nvp.numberofiterations;
if isempty(nIt)
    nIt = 999;
end
obj.trace.nfe = repmat(obj.trace.nfe, nIt, 1);

%% ======= MAIN LOOP =======  
for iIt = 1:nIt
% schedule posterior updates

    if bUpdateKmeans
        % initialize cluster parameters with kmeans
        obj = kmeans_init( obj );
        bUpdateKmeans = false;
    end

    if bUpdateAssignments
        % update alpha, q_nk
        obj = update_assigments( obj );
    end

    if bUpdateClusters
        if obj.options.confVar
            % update m_beta, S_beta
            obj = update_confounds( obj );
        end
        % update m_k, tau_k, nu_k, S_k
        obj = update_clusters( obj ); %%% TODO cluster-specific prior mean and scale
    end

    % update mu_n, Sigma_n, b_nr
    obj = update_dcm( obj, bUpdateClusters );

    % update negative free energy
    obj.trace.nfe(iIt) = obj.vb_nfe( );
    dNfe = obj.trace.nfe(iIt) - obj.posterior.nfe;
    obj.posterior.nfe = obj.trace.nfe(iIt);
    if bVerbose
        if isnan(dNfe)
            fprintf('iteration %3u.\n', iIt);
        else
            fprintf('iteration %3u, dF: %.2E.\n', iIt, dNfe);
        end
    end

    % check stopping conditions
    if ~bUpdateClusters
        % start updating clustering parameters and confound coefficients
        bUpdateClusters = (dNfe < obj.options.convergence.dDcm && dNfe >= 0);% || iIt > 32;
        if bUpdateClusters
            obj.trace.convergence(1) = iIt;
        end
        
    elseif ~bUpdateAssignments
        % start updating assignment labels
        bUpdateAssignments = dNfe < obj.options.convergence.dClusters;
        if bUpdateAssignments
            bUpdateKmeans = obj.options.convergence.bUseKmeans;
            obj.trace.convergence(2) = iIt;
            % force next iteration to happen
            obj.posterior.nfe = NaN;
        end
        
    elseif dNfe < obj.options.convergence.dVb
        % convergence of VB scheme
        obj.trace.convergence(3) = iIt;
        obj.trace.nfe(iIt+1:end) = [];
        obj.trace.epsilon = obj.aux.epsilon;
        if bVerbose
            fprintf('VB converged after %u iterations. ', iIt)
        end
        break;
        
    end
end

if bVerbose
    if iIt == nIt
        fprintf('VB reached the maximum number of iterations. ')
    end
    fprintf('Negative free energy: %.2E\n', obj.posterior.nfe);
end

%% post-processing
% estimate confound-free DCM parameters
if obj.options.confVar
    mu_beta = obj.aux.mu_beta;
    Sigma_beta = obj.aux.Sigma_beta;
    if obj.options.confVar > 1
        mu_beta = sum(bsxfun(@times, mu_beta, permute(obj.posterior.q_nk, ...
            [1 3 2])), 3);
        Sigma_beta = sum(bsxfun(@times, Sigma_beta, permute( ...
            obj.posterior.q_nk, [3 4 1 2])), 4);
    end
    obj.posterior.mu_r = obj.posterior.mu_n(:, 1:obj.idx.P_c) - mu_beta;
    obj.posterior.Sigma_r = obj.posterior.Sigma_n(1:obj.idx.P_c, 1:obj.idx.P_c, :) ...
        + Sigma_beta;
else    
    obj.posterior.mu_r = [];
    obj.posterior.Sigma_r = [];
end

% normalized variance of residual (by subject and region)
obj.posterior.nrv = zeros(obj.N, obj.R);
for n = 1:obj.N
    varRes = var(obj.aux.epsilon{n});
    varSig = var(obj.data(n).bold);
    obj.posterior.nrv(n,:) = varRes./varSig;
end

end


%% UPDATE: posterior over cluster parameters
function [ obj ] = update_clusters( obj )

% update parameters of inverse Wishart
obj.posterior.tau = obj.prior.tau_0 + obj.aux.q_k;
obj.posterior.nu  = obj.prior.nu_0  + obj.aux.q_k;

% auxiliary variables
Sigma_n_c = obj.posterior.Sigma_n(1:obj.idx.P_c,1:obj.idx.P_c,:);
mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c);
if obj.options.confVar == 1
    Sigma_n_c = Sigma_n_c + obj.aux.Sigma_beta;
    mu_n_c = mu_n_c - obj.aux.mu_beta;
end

for k = 1:obj.K

    if obj.options.confVar > 1
        Sigma_n_c = obj.posterior.Sigma_n(1:obj.idx.P_c,1:obj.idx.P_c,:) + ...
            obj.aux.Sigma_beta(:,:,:,k);
        mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c) - ...
            obj.aux.mu_beta(:,:,k);
    end
    % Eq (17)
    mu_k_c = obj.posterior.q_nk(:,k)'*mu_n_c;
    Sigma_k_c = sum(bsxfun(@times, Sigma_n_c, ...
        permute(obj.posterior.q_nk(:,k),[3,2,1])), 3);
    % Eq (16)
    % posterior cluster mean
    obj.posterior.m(k,:) = (mu_k_c + obj.prior.tau_0*obj.prior.m_0)/...
        (obj.aux.q_k(k) + obj.prior.tau_0);
    % posterior cluster covariance
    mu_k_c = mu_k_c./obj.aux.q_k(k);
    dmu = bsxfun(@minus, mu_n_c, mu_k_c);
    obj.posterior.S(:,:,k) = ...
        Sigma_k_c + ...
        dmu'*bsxfun(@times, dmu, obj.posterior.q_nk(:,k)) + ...
        obj.aux.q_k(k)*obj.prior.tau_0/...
            (obj.aux.q_k(k) + obj.prior.tau_0)*...
            (mu_k_c - obj.prior.m_0)'*(mu_k_c - obj.prior.m_0) + ...
        obj.prior.S_0;
    % update auxiliary variables
    obj.aux.ldS(k) = tapas_huge_logdet(obj.posterior.S(:,:,k));
    obj.aux.nu_inv_S(:,:,k) = obj.posterior.nu(k).*inv(obj.posterior.S(:,:,k));

end
end


%% UPDATE: posterior over cluster assignments
function [ obj ] = update_assigments( obj )

Sigma_n_c = obj.posterior.Sigma_n(1:obj.idx.P_c,1:obj.idx.P_c,:);
mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c);
if obj.options.confVar == 1
    Sigma_n_c = Sigma_n_c + obj.aux.Sigma_beta;
    mu_n_c = mu_n_c - obj.aux.mu_beta;
end
psiNu = sum(psi(0, bsxfun(@minus, obj.posterior.nu, 0:obj.idx.P_c-1)/2), 2);
psiAlpha = psi(0, obj.posterior.alpha);
log_q_nk = zeros(obj.N, obj.K);
% Eq (18)
for k = 1:obj.K

    if obj.options.confVar > 1
        Sigma_n_c = obj.posterior.Sigma_n(1:obj.idx.P_c,1:obj.idx.P_c,:) + ...
            obj.aux.Sigma_beta(:,:,:,k);
        mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c) - ...
            obj.aux.mu_beta(:,:,k);
    end
    dm = bsxfun(@minus,mu_n_c,obj.posterior.m(k,:));
    log_q_nk(:,k) = -.5*sum((dm*obj.aux.nu_inv_S(:,:,k)).*dm, 2) ...
        -.5*squeeze(sum(sum(bsxfun(@times, Sigma_n_c, ...
            obj.aux.nu_inv_S(:,:,k)), 1), 2));

end
log_q_nk = bsxfun(@plus,log_q_nk, (-.5*obj.aux.ldS +.5*psiNu ...
    -.5*obj.idx.P_c./obj.posterior.tau + psiAlpha)');
log_q_nk = exp(bsxfun(@minus, log_q_nk, max(log_q_nk, [], 2)));    
obj.posterior.q_nk = bsxfun(@rdivide, log_q_nk, sum(log_q_nk, 2));

% Eq (17)
obj.aux.q_k = sum(obj.posterior.q_nk,1)' + realmin;

% Eq (15)
obj.posterior.alpha = obj.prior.alpha_0 + obj.aux.q_k;

end


%% UPDATE: posterior over confound coefficients
function [ obj ] = update_confounds( obj )

m_k = permute(obj.posterior.m, [3,2,1]);
mu_n_c = obj.posterior.mu_n(:, 1:obj.idx.P_c);

for p = 1:obj.idx.P_c
    
    % Eq (XX)
    tmp = obj.aux.nu_inv_S(p, p, :);
    tmp = bsxfun(@times, obj.posterior.q_nk, tmp(:)');
    if obj.options.confVar == 1
        tmp = sum(tmp, 2);
    end
    % posterior precision over beta
    obj.posterior.S_beta(:, :, p, :) = bsxfun(@plus, obj.aux.Pi_beta_0, ...
        sum(bsxfun(@times, obj.aux.x_n_2, reshape(tmp, 1, 1, obj.N, [])), 3));

    % Eq (XX+1)
    obj.aux.mu_beta(:,p,:) = 0;    
    dmu = bsxfun(@minus, mu_n_c, obj.aux.mu_beta);
    dmu = bsxfun(@minus, dmu, m_k);
    tmp = permute(obj.posterior.q_nk, [1 3 2]).*...
        sum(bsxfun(@times, obj.aux.nu_inv_S(p, :, :), dmu), 2);
    if obj.options.confVar == 1
        tmp = sum(tmp, 3);
    end
    tmp = permute(sum(bsxfun(@times, tmp, obj.aux.x_n), 1), [2 1 3]);
    for k = 1:size(obj.posterior.m_beta, 3)        
        obj.posterior.m_beta(:,p,k) = obj.posterior.S_beta(:,:,p,k)\...
            (obj.aux.Pi_m_beta_0 + tmp(:,1,k));
        % posterior covariance over beta
        obj.posterior.S_beta(:,:,p,k) = inv(obj.posterior.S_beta(:,:,p,k));
        obj.aux.ldS_beta(p,k) = tapas_huge_logdet( ...
            obj.posterior.S_beta(:,:,p,k));        
    end
    % update auxiliary variables
    obj.aux.mu_beta(:,p,:) = sum(bsxfun(@times, obj.aux.x_n, ...
        permute(obj.posterior.m_beta(:,p,:), [2 1 3])), 2);
    obj.aux.Sigma_beta(p,p,:,:) = sum(sum(bsxfun(@times, obj.aux.x_n_2, ...
        obj.posterior.S_beta(:,:,p,:)), 1), 2);
    
end

end


%% UPDATE: posterior over DCM and noise parameters
function [ obj ] = update_dcm( obj, bCheckNfe )

% cache DCM parameters
tmpMu       = obj.posterior.mu_n;
tmpSigma    = obj.posterior.Sigma_n;
tmpLdSigma  = obj.aux.ldSigma;
tmpEpsilon  = obj.aux.epsilon;
tmpG        = obj.aux.G;
tmpB        = obj.posterior.b;
tmpBp       = obj.aux.b_prime;
tmpLb       = obj.aux.lambda_bar;
tmpNfe      = obj.vb_nfe( );
    
% subject-wise update DCM and noise parameters
for n = 1:obj.N
    varEps = sum(var(obj.aux.epsilon{n}));
    % Eq (20)
    m_k = obj.posterior.m;
    if obj.options.confVar
        tmp = permute(obj.aux.mu_beta(n,:,:), [3 2 1]);
        m_k = bsxfun(@plus, m_k, tmp);
    end
    lambda_prime_c = zeros(obj.idx.P_c);
    mu_prime_c = zeros(1,obj.idx.P_c);
    for k = 1:obj.K
        tmps = obj.posterior.q_nk(n,k)*obj.aux.nu_inv_S(:,:,k);
        lambda_prime_c = lambda_prime_c + tmps;            
        mu_prime_c = mu_prime_c + m_k(k,:)*tmps;
    end
    
 
    % Eq (19)
    % posterior covariance
    Lambda_bar = repmat(obj.aux.lambda_bar(n,:), obj.aux.q_r(n), 1);
    G_Lambda = bsxfun(@times, obj.aux.G{n}, Lambda_bar(:))';
    Pi_n = G_Lambda*obj.aux.G{n};
    Pi_n(1:obj.idx.P_c, 1:obj.idx.P_c) = lambda_prime_c + ...
        Pi_n(1:obj.idx.P_c, 1:obj.idx.P_c);
    Pi_n(obj.idx.P_c+1:end, obj.idx.P_c+1:end) = obj.aux.Pi_h + ...
        Pi_n(obj.idx.P_c+1:end, obj.idx.P_c+1:end);    
    obj.posterior.Sigma_n(:,:,n) = inv(Pi_n); %%% TODO Pi + diag(delta)
    obj.aux.ldSigma(n) = - tapas_huge_logdet(Pi_n);
    % posterior mean
    tmp = obj.aux.epsilon{n}(:) + obj.aux.G{n}*obj.posterior.mu_n(n,:)';
    % regularization factor
    tmp = tmp*(.9.^mod(obj.trace.nRetract(n),5));   
    obj.posterior.mu_n(n,:) = ...
        ((G_Lambda*tmp)' + [mu_prime_c, obj.aux.mu_prime_h])/Pi_n;
    % recalculate jacobian and residual
    obj = obj.options.fncBold(obj, n);
    
    % Eq (22)
    tmp = sum((obj.aux.G{n}*obj.posterior.Sigma_n(:,:,n)).*obj.aux.G{n},2);
    obj.aux.b_prime(n,:) = sum(obj.aux.epsilon{n}.^2 + ...
        reshape(tmp, obj.aux.q_r(n), obj.R), 1);  
    % Eq (21)
    obj.posterior.b(n,:) = obj.prior.b_0 + obj.aux.b_prime(n,:)/2;
    % Eq (23)
    obj.aux.lambda_bar(n,:) = obj.posterior.a(n,:)./obj.posterior.b(n,:);

    newNfe = obj.vb_nfe( );
    % check negative free energy (during first few iterations, accept
    % decrease in F if fit improves)
    bUnstable = any(isnan(obj.aux.G{n}(:))) || any(isinf(obj.aux.G{n}(:)));
    bOverride = bCheckNfe || varEps < sum(var(obj.aux.epsilon{n}));
    if bUnstable || (tmpNfe > newNfe && bOverride) 
        % retract update if new parameters unstable or if F decreased
        obj.posterior.mu_n(n,:)      = tmpMu(n,:);
        obj.posterior.Sigma_n(:,:,n) = tmpSigma(:,:,n);
        obj.aux.ldSigma(n)           = tmpLdSigma(n);
        obj.aux.epsilon{n}           = tmpEpsilon{n};
        obj.aux.G{n}                 = tmpG{n};
        obj.posterior.b(n,:)         = tmpB(n,:);
        obj.aux.b_prime(n,:)         = tmpBp(n,:);
        obj.aux.lambda_bar(n,:)      = tmpLb(n,:);
        obj.trace.nRetract(n)        = obj.trace.nRetract(n) + 1;
    else
        % otherwise accept update
        tmpNfe = newNfe;
        obj.trace.nDcmUpdate(n) = obj.trace.nDcmUpdate(n) + 1;
        obj.trace.nRetract(n)   = 0;
    end
    
end

end


%% initialize cluster parameters with K-means
function [ obj ] = kmeans_init( obj )
% kmeans
mu_n_c = obj.posterior.mu_n(:,1:obj.idx.P_c);
if obj.options.confVar
    mu_n_c = mu_n_c - obj.aux.mu_beta(:,:,1);
end
idx = kmeans(mu_n_c, obj.K, 'Replicates', obj.const.nKmeans);

% initialize additional clusters
obj.posterior.q_nk = zeros(obj.N,obj.K);
obj.posterior.q_nk(sub2ind([obj.N,obj.K], (1:obj.N)', idx)) = 1;
obj.aux.q_k = sum(obj.posterior.q_nk, 1)' + realmin;
obj.posterior.alpha = obj.prior.alpha_0 + obj.aux.q_k;
if obj.options.confVar > 1
    obj.posterior.m_beta = repmat(obj.posterior.m_beta(:,:,1), [1 1 obj.K]);
    obj.posterior.S_beta = repmat(obj.posterior.S_beta(:,:,:,1), ...
        [1 1 1 obj.K]);
    obj.aux.Sigma_beta = repmat(obj.aux.Sigma_beta(:,:,:,1), [1 1 1 obj.K]);
    obj.aux.ldS_beta = repmat(obj.aux.ldS_beta(:,1), 1, obj.K);
    obj.aux.mu_beta = repmat(obj.aux.mu_beta(:,:,1), [1 1 obj.K]);
end
obj = update_clusters( obj );

% debug info
obj.trace.kmeans.labels = idx;
obj.trace.kmeans.inputs = mu_n_c;

end



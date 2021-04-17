function [ obj ] = vb_init( obj )
% Initialize posterior and auxiliary variabels for VB inversion of HUGE.
% Requires obj.dcm and obj.prior to be intialized.
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



% method for gradient calculation 
obj.options.fncBold = @bold_grad_cd;

%% prior
prior = struct();
prior.alpha_0 = obj.prior.alpha_0;
prior.m_0 = obj.prior.m_0; % prior mean
prior.tau_0 = obj.prior.tau_0;
prior.nu_0 = max(obj.prior.nu_0, 1.5^obj.idx.P_c);
prior.nu_0 = min(prior.nu_0, double(realmax('single')));
% scaling matrix of inverse-Wishart
prior.S_0 = obj.prior.S_0*(prior.nu_0 - obj.idx.P_c - 1);
% prior mean of hemodynamic parameters
prior.mu_h = obj.prior.mu_h;
% prior Covariance of hemodynamic parameters
prior.Sigma_h = obj.prior.Sigma_h;
% prior inverse scale of observation noise (b_0 in Figure 1 of REF [1])
prior.b_0 = obj.prior.mu_lambda./obj.prior.s2_lambda;
% prior shape parameter of observation noise (a_0 in Figure 1 of REF [1])
prior.a_0 = obj.prior.mu_lambda.^2./obj.prior.s2_lambda;
% prior mean and covariance over confound coefficients
prior.m_beta_0 = obj.prior.m_beta_0;
prior.S_beta_0 = obj.prior.S_beta_0;

obj.prior = prior;

%% create posterior struct
obj.posterior = struct();
% cluster weights
obj.posterior.alpha = obj.prior.alpha_0;
obj.posterior.alpha(1) = obj.posterior.alpha(1) + obj.N;
% cluster parameters
obj.posterior.m = obj.options.start.clusters;
obj.posterior.tau = repmat(obj.prior.tau_0, obj.K, 1);
obj.posterior.tau(1) = obj.prior.tau_0 + obj.N;
obj.posterior.nu = repmat(obj.prior.nu_0, obj.K, 1);
obj.posterior.nu(1) = obj.prior.nu_0 + obj.N;
obj.posterior.S = repmat(obj.prior.S_0, 1, 1, obj.K);

% assigments
obj.posterior.q_nk = [ones(obj.N, 1),zeros(obj.N, obj.K - 1)];
% DCM parameters
obj.posterior.mu_n = obj.options.start.subjects;
Sigma = zeros(obj.idx.P_c + obj.idx.P_h);
Sigma(1:obj.idx.P_c, 1:obj.idx.P_c) = obj.prior.S_0/(obj.prior.nu_0 - obj.idx.P_c - 1);
Sigma(1:obj.idx.P_c, 1:obj.idx.P_c) = obj.prior.S_0;
Sigma(obj.idx.P_c+1:end, obj.idx.P_c+1:end) = obj.prior.Sigma_h;
obj.posterior.Sigma_n = repmat(Sigma, 1, 1, obj.N);
% noise
obj.posterior.b = repmat(obj.prior.b_0, obj.N, obj.R);
obj.posterior.a = repmat(obj.prior.a_0, obj.N, obj.R);
% confounds
if obj.options.confVar
    if obj.options.confVar > 1
        obj.options.confVar = obj.K;
    end
    obj.posterior.m_beta = repmat(obj.prior.m_beta_0, 1, obj.idx.P_c, ...
        obj.options.confVar);
    obj.posterior.S_beta = repmat(obj.prior.S_beta_0, 1, 1, obj.idx.P_c, ...
        obj.options.confVar);    
else
    obj.posterior.m_beta = [];
    obj.posterior.S_beta = [];
end


%% auxiliary variables
obj.aux.epsilon = cell(obj.N, 1);   % residual
obj.aux.G       = cell(obj.N, 1);   % jacobian
if obj.options.confVar    % confounds
    % center and normalize confound
    obj.aux.x_n = [obj.data(:).confounds]';
    obj.aux.x_n = bsxfun(@minus, obj.aux.x_n, mean(obj.aux.x_n));
    tmp = std(obj.aux.x_n);
    tmp(tmp < eps) = 1;
    obj.aux.x_n = bsxfun(@rdivide, obj.aux.x_n, tmp);
    % confound-related auxiliary variabels
    obj.aux.x_n_2 = zeros(obj.M, obj.M, obj.N);
    tmp = zeros(obj.idx.P_c, obj.idx.P_c, obj.N);
    for n = 1:obj.N
        obj.aux.x_n_2(:,:,n) = obj.aux.x_n(n,:)'*obj.aux.x_n(n,:);
        tmp(:,:,n) = obj.aux.x_n(n,:)*obj.prior.S_beta_0*obj.aux.x_n(n,:)'...
            *eye(obj.idx.P_c);
    end
    obj.aux.Sigma_beta = repmat(tmp, 1, 1, 1, obj.options.confVar);
    obj.aux.Pi_beta_0 = inv(obj.prior.S_beta_0);
    obj.aux.Pi_m_beta_0 = obj.aux.Pi_beta_0*obj.prior.m_beta_0;
    obj.aux.ldS_beta = repmat(tapas_huge_logdet(obj.prior.S_beta_0), ...
        obj.idx.P_c, obj.options.confVar);
    obj.aux.mu_beta = zeros(obj.N, obj.idx.P_c, obj.options.confVar);
    for k = 1:obj.options.confVar
        obj.aux.mu_beta(:,:,k) = obj.aux.x_n*obj.posterior.m_beta(:,:,k);
    end
end

obj.aux.q_k = sum(obj.posterior.q_nk, 1)' + realmin;
obj.aux.ldS = repmat(tapas_huge_logdet(obj.prior.S_0), obj.K, 1);
obj.aux.ldSigma = repmat(tapas_huge_logdet(Sigma), obj.N ,1);
obj.aux.mu_prime_h = obj.prior.mu_h/obj.prior.Sigma_h;
obj.aux.Pi_h = inv(obj.prior.Sigma_h);
% nu_k*inv(S_k)
obj.aux.nu_inv_S = zeros(size(obj.posterior.S));
for k = 1:obj.K
    obj.aux.nu_inv_S(:,:,k) = obj.posterior.nu(k).*...
        inv(obj.posterior.S(:,:,k));
end
obj.aux.b_prime = zeros(obj.N, obj.R);

%% check stability of starting values
obj.aux.q_r = zeros(obj.N, 1);
for n = 1:obj.N
    obj.aux.q_r(n) = size(obj.data(n).bold, 1);
    obj = obj.options.fncBold(obj, n);
    assert(~(any(isnan(obj.aux.G{n}(:))) || any(isinf(obj.aux.G{n}(:)))), ...
        'TAPAS:HUGE:Init:Stability',...
        'Starting values for subject %u lead to instable DCM.', n);

    tmp = sum((obj.aux.G{n}*obj.posterior.Sigma_n(:,:,n)).*obj.aux.G{n}, 2);
    obj.aux.b_prime(n,:) = sum(obj.aux.epsilon{n}.^2 + ...
        reshape(tmp, obj.aux.q_r(n), obj.R), 1);    
    obj.posterior.b(n,:) = obj.prior.b_0 + obj.aux.b_prime(n,:)/2;
    obj.posterior.a(n,:) = obj.prior.a_0 + obj.aux.q_r(n)/2;
end


%% initialize auxiliary variables
obj.aux.lambda_bar = obj.posterior.a./obj.posterior.b;

obj.aux.mu_k_c(1,:) = mean(obj.posterior.mu_n(:, 1:obj.idx.P_c), 1);
obj.aux.Sigma_k_c(:,:,1) = ...
    sum(obj.posterior.Sigma_n(1:obj.idx.P_c, 1:obj.idx.P_c, :), 3);

% offline part of negative free energy
F_off = -obj.K*(obj.idx.P_c*(obj.idx.P_c - 1)/4*log(pi) + ...
    sum(gammaln((obj.prior.nu_0-obj.idx.P_c+1:obj.prior.nu_0)/2)));
F_off = F_off +.5*obj.K*obj.prior.nu_0*tapas_huge_logdet(obj.prior.S_0);
F_off = F_off +.5*obj.K*obj.idx.P_c*log(obj.prior.tau_0);
F_off = F_off + gammaln(sum(obj.prior.alpha_0));
F_off = F_off - sum(gammaln(obj.prior.alpha_0));
F_off = F_off - gammaln(obj.N + sum(obj.prior.alpha_0));
F_off = F_off - obj.N*obj.R*gammaln(obj.prior.a_0);
F_off = F_off + sum(gammaln(obj.posterior.a(:)));
F_off = F_off + obj.N*obj.R*obj.prior.a_0*log(obj.prior.b_0);
F_off = F_off -.5*obj.N*tapas_huge_logdet(obj.prior.Sigma_h);
F_off = F_off -.5*obj.idx.P_c*tapas_huge_logdet(obj.prior.S_beta_0);
F_off = F_off +.5*obj.N*obj.idx.P_c*log(2);
F_off = F_off -.5*sum(obj.aux.q_r)*obj.R*log(2*pi);
F_off = F_off +.5*obj.N*(obj.idx.P_c + obj.idx.P_h);
F_off = F_off +.5*obj.idx.P_c*(obj.K*(1 + obj.prior.nu_0) + obj.N);
if obj.options.confVar
    F_off = F_off +.5*obj.M*obj.idx.P_c*obj.options.confVar;    
end
obj.aux.F_off = F_off;

% evaluate negative free energy (VB)
obj.posterior.nfe = obj.vb_nfe( );


%% initialize trace
obj.trace = struct();
obj.trace.nfe = NaN;
obj.trace.nDcmUpdate = zeros(obj.N, 1);
obj.trace.nRetract   = zeros(obj.N,1);

end



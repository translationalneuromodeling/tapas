function [ obj ] = mh_invert( obj )
% Metropolized Gibbs sampling on collapsed HUGE model.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
% 
% 

obj = obj.mh_init( );
% number of iterations and burn-in
nIt = obj.options.nvp.numberofiterations;
if isempty(nIt)
    nIt = 2e5;
end
nBi = obj.options.nvp.burnin;
if isempty(nBi) || nBi >= nIt
    nBi = fix(nIt/2);
end

% inverse chain temperature
invTemp = obj.options.nvp.inversetemperature;

% reserve memory
q_nk = zeros(obj.N, obj.K);
obj.trace = struct();
obj.trace.smp = repmat(obj.aux.sample, nIt, 1);
obj.trace.lpr = repmat(obj.aux.lpr, nIt, 1);
psrf = repmat(obj.aux.sample, fix(nIt/obj.const.nPsrf) + 1, 1);

%% ===== MAIN LOOP =====
for iIt = 1:nIt
    
    % subject level
    for iMhDcm = 1:obj.options.mh.nSteps.dcm
        % sample DCM (parameters)
        obj = mh_sample_dcm( obj, invTemp );
        % sample lambda (hyperparameters)
        obj = mh_sample_noise( obj, invTemp );  
    end

    % group level
    if mod(iIt, obj.options.mh.nSteps.knKm) == 0
        k = mod(iIt/obj.options.mh.nSteps.knKm - 1,obj.K) + 1;
        obj = mh_sample_kmhop( obj, k );
    else
        % sample weights pi
        if obj.K > 1
            for iMh = 1:obj.options.mh.nSteps.weights
                obj = mh_sample_weights( obj );
            end
        end
        % sample cluster mu and Sigma
        for iMh = 1:obj.options.mh.nSteps.clusters
            obj = mh_sample_cluster( obj );
        end
    end
    
    % save sample
    obj.trace.smp(iIt) = obj.aux.sample;
    obj.trace.lpr(iIt) = obj.aux.lpr;

    % adapt proposal step size
    if (iIt <= nBi/3) && (mod(iIt, obj.const.mhAdapt(1)) == 0)
        obj = mh_adapt(obj, iIt);
    end
    
    % accumulate cluster assigment estimate
    if iIt > nBi
        tmp = bsxfun(@times, exp(obj.aux.rho), obj.aux.sample.pi);
        q_nk = q_nk + bsxfun(@rdivide, tmp, sum(tmp, 2));
    end
    % convergence monitoring
    if mod(iIt, obj.const.nPsrf) == 0
        iPsrf = iIt/obj.const.nPsrf;
        psrf(iPsrf) = mh_psrf(obj.trace.smp(1:iIt), 4);
    end
    if obj.options.nvp.verbose && mod(iIt, 100) == 1
        fprintf('Iteration %u\n', iIt);
    end
    
% -------------------------------------
end%         END MAIN LOOP
% -------------------------------------
%%          Post Processing
% estimates for assignment probability
q_nk = q_nk/(nIt - nBi);

% acceptance ratio
ratio = struct();
obj.aux.nProp.sp(1) = obj.aux.nProp.sp(1)*obj.N;
for parameter = {'pi', 'mu', 'kappa', 'theta', 'lambda', 'sp'}
    ratio.(parameter{1}) = obj.aux.nAccept.(parameter{1})./...
        obj.aux.nProp.(parameter{1});
end
% posterior mean and quantiles
postMean    = struct();
postVar     = struct();
postQuant   = struct();
for parameter = {'pi', 'mu', 'kappa', 'theta_c', 'theta_h', 'lambda'}
    tmp = reshape([obj.trace.smp(nBi+1:end).(parameter{1})], ...
        [size(obj.aux.sample.(parameter{1})), nIt - nBi]);
    postMean.(parameter{1})  = mean(tmp, 3);
    postVar.(parameter{1})   = var(tmp, 0, 3);
    postQuant.(parameter{1}) = quantile(tmp, obj.options.quantiles, 3);
end
% cumulative probability levels for quantiles
postQuant.levels = obj.options.quantiles;

% calculate PSRF post burn-in
psrf(end) = mh_psrf(obj.trace.smp(nBi + 1:end), 4);

% collect posterior summaries
obj.posterior = struct('nIt', nIt, 'nBi', nBi, 'q_nk', q_nk, ...
    'ratio', ratio, 'psrf', psrf, 'mean', postMean, 'variance', postVar, ...
    'quantile', postQuant, 'lvarBold', obj.aux.lvarBold, ...
    'nrv', exp(-postMean.lambda));

% thin MC chain
if ~isempty(obj.options.nTrace)
    % keep only nTrace samples from post-burn-in phase
    nTrace  = min(nIt - nBi, obj.options.nTrace);
    nThin   = fix((nIt - nBi)/nTrace);
    % select samples uniformly
    obj.trace.smp = obj.trace.smp(end-nThin*(nTrace - 1):nThin:end);
    obj.trace.lpr = obj.trace.lpr(end-nThin*(nTrace - 1):nThin:end);
end

end

%---------------------------------
%            SAMPLING
%---------------------------------
%% SAMPLING: weights (pi)
function [ obj ] = mh_sample_weights( obj )

% propose (in unconstrained space) 
prop_piu = obj.aux.sample.pi_u + ...
    randn(1,obj.K-1)*obj.aux.step.pi;
% transform to unit simplex
prop_pis = 1./(1 + exp(log(obj.K-1:-1:1) - prop_piu));
prop_pic = cumprod(1-prop_pis);
prop_pi = [prop_pis(1),-diff(prop_pic),prop_pic(end)];

% evaluate joint (in unconstrained space)
% log-prior on pi
prop_lpr = log(prop_pi)*(obj.prior.alpha_0 - 1) + ...
   sum(log(prop_pis)) + sum(log(1-prop_pis)) + ...
   sum(log(prop_pic(1:end-1)));
prop_lpr = max(prop_lpr, -realmax);
% log-conditional of theta_c given pi
prop_lcd = log(exp(obj.aux.rho)*prop_pi') + obj.aux.rho_max;

% accept/reject
obj.aux.nProp.pi = obj.aux.nProp.pi + 1;
a = exp(prop_lpr - obj.aux.lpr.pi...
        + sum(prop_lcd - obj.aux.lpr.theta_c));
%         a = exp(prop_lpr - obj.aux.lpr.pi);
if ~isnan(a) && ~isinf(a) && rand()<a
    obj.aux.sample.pi_u = prop_piu;
    obj.aux.sample.pi = prop_pi;
    obj.aux.lpr.pi = prop_lpr;
    obj.aux.lpr.theta_c = prop_lcd;
    obj.aux.nAccept.pi = obj.aux.nAccept.pi + 1;
end

end


%% SAMPLING: cluster parameters (mu, kappa = - log(sigma^2))
function [ obj ] = mh_sample_cluster( obj )

obj.aux.nProp.mu = obj.aux.nProp.mu + 1;
obj.aux.nProp.kappa = obj.aux.nProp.mu;

for k = 1:obj.K

    prop_kappa = obj.aux.sample.kappa(k,:);
    % mean
    tmp = randn(1,obj.idx.P_c)*obj.aux.step.mu(k);
    prop_mu = obj.aux.sample.mu(k,:) + tmp*...
        obj.aux.transform.mu(:,:,k);
    dlq = 0; % delta log-proposal density

    % evaluate log-conditional
    % prior
    prop_dmu_k = prop_mu - obj.prior.m_0;
    prop_lpr_mu = -prop_dmu_k*obj.prior.T_0*prop_dmu_k'/2;

    % log-conditional theta_c given mu and kappa
    prop_dtheta_c = bsxfun(@minus, obj.aux.sample.theta_c, prop_mu);
    tmp = obj.aux.l2pi - .5*prop_dtheta_c.^2*exp(prop_kappa') ...
        + .5*sum(prop_kappa);
    if obj.K > 1
        prop_rho = bsxfun(@plus, obj.aux.rho, obj.aux.rho_max);
        prop_rho(:,k) = tmp;
        prop_rho_max = max(prop_rho, [], 2);
        prop_rho = bsxfun(@minus, prop_rho, prop_rho_max);
        prop_lcd = log(exp(prop_rho)*obj.aux.sample.pi') ...
            + prop_rho_max; % log-sum-exp
    else
        prop_rho_max = tmp;
        prop_lcd = prop_rho_max;
        prop_rho = zeros(obj.N, 1);
    end

    % accept/reject
    a = exp(sum(prop_lcd - obj.aux.lpr.theta_c) ...
          + prop_lpr_mu - obj.aux.lpr.mu(k) + dlq);
%             a = exp(prop_lpr_mu - obj.aux.lpr.mu(k));
    if ~isnan(a) && ~isinf(a) && rand()<a
        obj.aux.nAccept.mu(k) = obj.aux.nAccept.mu(k) + 1;
        obj.aux.sample.mu(k,:) = prop_mu;
        obj.aux.rho = prop_rho;
        obj.aux.rho_max = prop_rho_max;
        obj.aux.lpr.theta_c = prop_lcd;
        obj.aux.lpr.mu(k) = prop_lpr_mu;
    end

    % precision
    prop_kappa = prop_kappa + randn(1,obj.idx.P_c)*obj.aux.step.kappa(k);

    % evaluate log-conditional
    % prior
    prop_dkappa = prop_kappa - obj.prior.s_0;
    prop_lpr_kappa = -.5*prop_dkappa.^2*obj.prior.nu_0;
    % log-conditional theta_c given mu and kappa
    prop_dtheta_c = bsxfun(@minus, obj.aux.sample.theta_c, ...
        obj.aux.sample.mu(k,:));
    tmp = obj.aux.l2pi - .5*prop_dtheta_c.^2*exp(prop_kappa') ...
        + .5*sum(prop_kappa);
    if obj.K > 1
        prop_rho = bsxfun(@plus, obj.aux.rho, obj.aux.rho_max);
        prop_rho(:,k) = tmp;
        prop_rho_max = max(prop_rho, [], 2);
        prop_rho = bsxfun(@minus, prop_rho, prop_rho_max);
        prop_lcd = log(exp(prop_rho)*obj.aux.sample.pi') ...
            + prop_rho_max;
    else
        prop_rho_max = tmp;
        prop_lcd = prop_rho_max;
        prop_rho = zeros(obj.N, 1);
    end

    % accept/reject
    a = exp(sum(prop_lcd - obj.aux.lpr.theta_c) ...
          + prop_lpr_kappa - obj.aux.lpr.kappa(k));
%             a = exp(prop_lpr_kappa - obj.aux.lpr.kappa(k));
    if ~isnan(a) && ~isinf(a) && rand()<a
        obj.aux.nAccept.kappa(k) = obj.aux.nAccept.kappa(k) + 1;
        obj.aux.sample.kappa(k,:) = prop_kappa;
        obj.aux.rho = prop_rho;
        obj.aux.rho_max = prop_rho_max;
        obj.aux.lpr.theta_c = prop_lcd;
        obj.aux.lpr.kappa(k) = prop_lpr_kappa;
    end

end

end


%% SAMPLING: DCM parameter (theta)
function [ obj ] = mh_sample_dcm( obj, invTemp )
nProp = obj.aux.nProp.theta + obj.aux.nProp.sp(1) + 1;
bGmm = mod(nProp, obj.options.mh.nSteps.knGmm) == 0;
if bGmm
    obj.aux.nProp.sp(1) = obj.aux.nProp.sp(1) + 1;
else
    obj.aux.nProp.theta = obj.aux.nProp.theta + 1;
end
prec = -.5*exp(obj.aux.sample.kappa);

for n = 1:obj.N

    % propose
    if bGmm % sample from GMM        
        k = randsample(obj.K, 1, true, obj.aux.sample.pi);
        prop_theta_c = obj.aux.sample.mu(k,:) + ...
            randn(1, obj.idx.P_c).*exp(-.5*obj.aux.sample.kappa(k,:));
        prop_theta_h = obj.prior.mu_h + randn(1,obj.idx.P_h)./...
            sqrt(diag(obj.prior.Pi_h))'; % assume diagonal precision
    else % sample from Gaussian kernel
%             prop_theta_c = obj.aux.sample.theta_c(n,:) + ...
%                 randn(1, obj.idx.P_c).*obj.aux.step.theta_c(n,:);
        tmp = (randn(1, obj.idx.P_c + obj.idx.P_h).*obj.aux.step.theta(n,:))* ...
            obj.aux.transform.theta;
        prop_theta_c = obj.aux.sample.theta_c(n,:) + tmp(1:obj.idx.P_c);
        prop_theta_h = obj.aux.sample.theta_h(n,:) + tmp(obj.idx.P_c+1:end);
    end

    % evaluate joint
    % log-prior
    prop_dtheta_c = bsxfun(@minus, prop_theta_c, obj.aux.sample.mu);
    prop_rho = obj.aux.l2pi + sum(prec.*prop_dtheta_c.^2, 2)' ...
        + 0.5*sum(obj.aux.sample.kappa, 2)';
    prop_rho_max = max(prop_rho);
    prop_rho = prop_rho - prop_rho_max;
    prop_lpr_c = log(exp(prop_rho)*obj.aux.sample.pi(:)) + prop_rho_max;            
    prop_dtheta_h = prop_theta_h - obj.prior.mu_h;
    prop_lpr_h = -.5*prop_dtheta_h*obj.prior.Pi_h*prop_dtheta_h';
    % log-likelihood
    if invTemp
        prop_epsilon = obj.bold_gen( [prop_theta_c, prop_theta_h], ...
            obj.data(n), obj.inputs(n), obj.options.hemo, obj.R, ...
            obj.L, obj.idx );
        tmp = obj.aux.sample.lambda(n,:) - obj.aux.lvarBold(n,:);
        prop_llh = -.5*sum(prop_epsilon.^2*exp(tmp)') ...
           +.5*obj.aux.q_r(n)*sum(tmp);
    else % skip calculation of BOLD signal if inverse temp is zero
        prop_llh = 0;
        prop_epsilon = 0;
    end

    % accept/reject
    a = exp(~bGmm*(prop_lpr_c - obj.aux.lpr.theta_c(n) ...
        + prop_lpr_h - obj.aux.lpr.theta_h(n)) ...
        + invTemp*(prop_llh - obj.aux.lpr.llh(n)));

    if ~isnan(a) && ~isinf(a) && rand()<a
        if bGmm
            obj.aux.nAccept.sp(1) = obj.aux.nAccept.sp(1) + 1;
        else
            obj.aux.nAccept.theta(n) = obj.aux.nAccept.theta(n) + 1;
        end
        obj.aux.sample.theta_c(n,:) = prop_theta_c;
        obj.aux.sample.theta_h(n,:) = prop_theta_h;
        obj.aux.lpr.theta_c(n) = prop_lpr_c;
        obj.aux.lpr.theta_h(n) = prop_lpr_h;
        obj.aux.rho(n,:) = prop_rho;
        obj.aux.rho_max(n) = prop_rho_max;
        obj.aux.epsilon{n} = prop_epsilon;
        obj.aux.lpr.llh(n) = prop_llh;
    end

end
end


%% SAMPLING: Signal-to-noise (lambda)
function [ obj ] = mh_sample_noise( obj, invTemp )

obj.aux.nProp.lambda = obj.aux.nProp.lambda + 1;
for n = 1:obj.N

    % propose in unconstrained space
    prop_lambda = obj.aux.sample.lambda(n,:) + ...
        randn(1, obj.R).*obj.aux.step.lambda(n);

    % evaluate log-conditional
    % prior
    prop_dlambda = prop_lambda - obj.prior.lambda_0;
    prop_lpr = -.5*prop_dlambda.^2*obj.prior.omega_0;
    % likelihood
    tmp = prop_lambda - obj.aux.lvarBold(n,:);
    prop_llh = -.5*sum(obj.aux.epsilon{n}.^2*exp(tmp)') ...            
        +.5*obj.aux.q_r(n)*sum(tmp);

    % accept/reject
    a = exp(prop_lpr - obj.aux.lpr.lambda(n) ...
            + invTemp*(prop_llh - obj.aux.lpr.llh(n)));

    if ~isnan(a) && ~isinf(a) && rand()<a
        obj.aux.nAccept.lambda(n) = obj.aux.nAccept.lambda(n) + 1;
        obj.aux.sample.lambda(n,:) = prop_lambda;
        obj.aux.lpr.lambda(n) = prop_lpr;
        obj.aux.lpr.llh(n) = prop_llh;
    end
end
end


%% SAMPLING: k-means-based mode hopping
% A special proposal for clustering part of HUGE model (pi, mu and kappa)
function [ obj ] = mh_sample_kmhop(obj, k)
% track acceptance rate
obj.aux.nProp.sp(2) = obj.aux.nProp.sp(2) + 1;

% --- do kmeans for k = 1,...,K, and define q(...|k) ---
% set up prior
tmpm = exp(obj.prior.s_0+1./2./obj.prior.nu_0');
tmpv = (exp(1./obj.prior.nu_0')-1).*exp(2*obj.prior.s_0+1./obj.prior.nu_0');
b0 = tmpm./tmpv;
a0 = tmpm.^2./tmpv;

% get data
X = obj.aux.sample.theta_c;

% reserve memory
qmm = repmat(obj.prior.m_0,obj.K,1,obj.K);
qms = repmat(diag(obj.prior.T_0)',obj.K,1,obj.K);
qkm = repmat(obj.prior.s_0,obj.K,1,obj.K);
qks = repmat(obj.prior.nu_0',obj.K,1,obj.K);
qpm = zeros(obj.K,obj.K);
for k1 = 1:obj.K
    % do kmeans
    idx = kmeans(X,k1,'Replicates',100);
    % assemble parameters for q
    for k2 = 1:obj.K
        nk = nnz(idx(:)==k2);
        if nk > 0
            mk = mean(X(idx(:)==k2,:),1);
            ak = a0 + (nk + 1)/2;
            bk = b0 + 0.5.*sum(bsxfun(@minus,X(idx(:)==k2,:),mk).^2,1);
            tmpm = ak./bk;
            tmpv = ak./bk.^2;
            % q(kappa)
            qks(k2,:,k1) = 1./log(tmpv./tmpm.^2 + 1);
            qkm(k2,:,k1) = log(tmpm) - 1./2./qks(k2,:,k1);
            % q(mu)
            qms(k2,:,k1) = diag(obj.prior.T_0)' + nk.*exp(obj.prior.s_0);
            qmm(k2,:,k1) = (diag(obj.prior.T_0)'.*obj.prior.m_0 + ...
                nk.*exp(obj.prior.s_0).*mk)./qms(k2,:,k1);
        end
        qpm(k1,k2) = nk + obj.prior.alpha_0(k2);
    end
end

% --- generate proposal for k in 1,...,K, and eval p ---
k1 = k;

prop_lpr_mu = zeros(1,obj.K);
prop_lpr_kappa = zeros(1,obj.K);
prop_rho = obj.aux.rho;

prop_pi = zeros(1,obj.K);
prop_kappa = zeros(obj.K,obj.idx.P_c);
prop_mu = zeros(obj.K,obj.idx.P_c);
for k2 = 1:obj.K
    % draw kappa*
    prop_kappa(k2,:) = qkm(k2,:,k1) + randn(1,obj.idx.P_c)./sqrt(qks(k2,:,k1));
    % draw mu*
    prop_mu(k2,:) = qmm(k2,:,k1) + randn(1,obj.idx.P_c)./sqrt(qms(k2,:,k1));
    % draw pi*
    prop_pi(1,k2) = gamrnd(qpm(k1,k2),1);
    
    % eval log p(mu*)
    prop_dmu_k = prop_mu(k2,:) - obj.prior.m_0;
    prop_lpr_mu(k2) = -prop_dmu_k*obj.prior.T_0*prop_dmu_k'/2;
    % eval log p(kappa*)
    prop_dkappa = prop_kappa(k2,:) - obj.prior.s_0;
    prop_lpr_kappa(k2) = -.5*prop_dkappa.^2*obj.prior.nu_0;
    
    % log-conditional theta_c given mu and kappa
    prop_dtheta_c = bsxfun(@minus, obj.aux.sample.theta_c, prop_mu(k2,:));
    prop_rho(:,k2) = obj.aux.l2pi - .5*prop_dtheta_c.^2*exp(prop_kappa(k2,:)') ...
        + .5*sum(prop_kappa(k2,:));

end
prop_pi = prop_pi./sum(prop_pi);

% eval log p(pi*)
prop_pis = [prop_pi(1),prop_pi(2:end-1)./(1 - cumsum(prop_pi(1:end-2)))];
prop_pic = cumprod(1-prop_pis);
prop_piu = tapas_huge_logit(prop_pis) - log(1./(obj.K-1:-1:1));
% log-prior on pi
prop_lpr_pi = log(prop_pi)*(obj.prior.alpha_0 - 1) + ...
   sum(log(prop_pis)) + sum(log(1-prop_pis)) + ...
   sum(log(prop_pic(1:end-1)));
prop_lpr_pi = max(prop_lpr_pi, -realmax);

% eval log p(theta|pi*,mu*,kappa*)
prop_rho_max = max(prop_rho, [], 2);
prop_rho = bsxfun(@minus, prop_rho, prop_rho_max);
prop_lcd = log(exp(prop_rho)*prop_pi') + prop_rho_max; % log-sum-exp

% --- iterate over all permutations and eval q ---
% eval log q(pi*,kappa*,mu*)
[prop_lq_pi,prop_lq_kappa,prop_lq_mu] = eval_lq_perm(prop_pi,...
    prop_kappa,prop_mu,qpm,qkm,qks,qmm,qms,obj.K);

% eval log q(pi,kappa,mu)
[smp_lq_pi,smp_lq_kappa,smp_lq_mu] = eval_lq_perm(obj.aux.sample.pi,...
    obj.aux.sample.kappa,obj.aux.sample.mu,qpm,qkm,qks,qmm,qms,obj.K);

m_prop_lq_mu = logmeanexp(prop_lq_mu);
m_prop_lq_kappa = logmeanexp(prop_lq_kappa);
m_prop_lq_pi = logmeanexp(prop_lq_pi);

m_smp_lq_mu = logmeanexp(smp_lq_mu);
m_smp_lq_kappa = logmeanexp(smp_lq_kappa);
m_smp_lq_pi = logmeanexp(smp_lq_pi);

% --- eval MH acceptance ratio ---
a = [sum(prop_lcd - obj.aux.lpr.theta_c) ...
    ;sum(prop_lpr_kappa - obj.aux.lpr.kappa) ...
    ;sum(prop_lpr_mu - obj.aux.lpr.mu) ...
    ;prop_lpr_pi - obj.aux.lpr.pi ...
    ;sum(m_smp_lq_kappa - m_prop_lq_kappa) ...
    ;sum(m_smp_lq_mu - m_prop_lq_mu) ...
    ;m_smp_lq_pi - m_prop_lq_pi ...
];
% figure(1);clf;stem(a);
a = min(1,exp(sum(a)));

% accept/reject
if ~isnan(a) && ~isinf(a) && rand()<a
    obj.aux.nAccept.sp(2) = obj.aux.nAccept.sp(2) + 1;
    obj.aux.sample.pi_u = prop_piu;
    obj.aux.sample.pi = prop_pi;
    obj.aux.sample.mu = prop_mu;
    obj.aux.sample.kappa = prop_kappa;
    obj.aux.lpr.pi = prop_lpr_pi;
    obj.aux.lpr.mu = prop_lpr_mu;
    obj.aux.lpr.kappa = prop_lpr_kappa;
    obj.aux.lpr.theta_c = prop_lcd;
    obj.aux.rho = prop_rho;
    obj.aux.rho_max = prop_rho_max;
end

end

% evaluate log q(pi*,kappa*,mu*) for all permutations
function [lq_pi,lq_kappa,lq_mu] = eval_lq_perm(p_pi,p_kappa,p_mu,qpm,qkm,qks,qmm,qms,K)
pidx = perms(1:K);

lq_mu = zeros(size(pidx));
lq_kappa = zeros(size(pidx));
lq_pi = zeros(size(pidx));

x = p_pi;
z = [x(1),x(2:end-1)./(1 - cumsum(x(1:end-2)))];
c = cumprod(1-z);

for k1 = 1:K
    m_pi = zeros(K,K);
    m_mu = zeros(size(m_pi));
    m_kappa = zeros(size(m_pi));
    % mesh log q(pi)
    tmp = gammaln(sum(qpm(k1,:))) - sum(gammaln(qpm(k1,:))) + ... % const
    sum(log(z)) + sum(log(1-z)) + ... % jacobian
    sum(log(c(1:end-1))); % jacobian
    m_pi(:,:) = (qpm(k1,:)' - 1)*log(x(:)') + tmp/K; % x

    % mesh log q(mu,kappa)
    for k2 = 1:K
        for k3 = 1:K
            % log p(mu*)
            m_mu(k2,k3) = -.5*(p_mu(k3,:) - qmm(k2,:,k1)).^2*qms(k2,:,k1)' -.5*log(2*pi) +.5*sum(log(qms(k2,:,k1)));
            % log p(kappa*)
            m_kappa(k2,k3) = -.5*(p_kappa(k3,:) - qkm(k2,:,k1)).^2*qks(k2,:,k1)' -.5*log(2*pi) +.5*sum(log(qks(k2,:,k1)));
        end
    end
    
    linInd = sub2ind([K,K], repmat(1:K,size(pidx,1),1), pidx);
    lq_mu(:,k1) = sum(m_mu(linInd),2);
    lq_kappa(:,k1) = sum(m_kappa(linInd),2);
    lq_pi(:,k1) = sum(m_pi(linInd),2);

end
end

% log(mean(exp(log_probability)))
function lmep = logmeanexp(lpr)
tmp = max(lpr(:));
lmep = log(mean(exp(lpr(:) - tmp))) + tmp;
end


%---------------------------------
%              MISC
%---------------------------------
%% adapt proposal distribution
function [ obj ] = mh_adapt(obj, iIt)

% adapt step size for:
for parameter = {'pi', 'mu', 'kappa', 'theta', 'lambda'}
    % current acceptance rate
    ratio = obj.aux.nAccept.(parameter{1})./obj.aux.nProp.(parameter{1});
    % correction factor
    tmp = ratio - obj.const.mhRate;
    tmp = exp(sign(tmp).*abs(tmp).^3.*obj.const.mhReg);
    % adapt step size
    obj.aux.step.(parameter{1}) = obj.aux.step.(parameter{1}).*tmp;
    % reset counter
    obj.aux.nProp.(parameter{1}) = 0;
    obj.aux.nAccept.(parameter{1}) = ...
        zeros(size(obj.aux.nAccept.(parameter{1})));

end

step = fix(iIt/obj.const.mhTrans);
idx = iIt:-step:1;

% match proposal distribution to posterior covariance for ...
% ... cluster means
mu = reshape([obj.trace.smp(idx).mu], obj.K, obj.idx.P_c, []);
for k = 1:obj.K
    % calculate SVD on empirical covariance of posterior samples
    tmp = cov(permute(mu(k,:,:), [3 2 1]));
    [rotation, scales] = svd(tmp);
    scales = sqrt(diag(scales));
    % limit smallest proposal step size to 5% of maximum step size
    scales(scales < max(scales)*.05) = max(scales)*.05;
    if max(scales) > 0
        obj.aux.transform.mu(:,:,k) = bsxfun(@times, rotation, scales')';
        tmp = sum(log(scales));
        obj.aux.step.mu(k) = obj.aux.step.mu(k)*exp(...
            (obj.aux.logdet.mu(k) - tmp)/obj.idx.P_c);
        obj.aux.logdet.mu(k) = tmp;        
    end
end

% ... subject means
theta = [reshape([obj.trace.smp(idx).theta_c], obj.N, obj.idx.P_c, []), ...
         reshape([obj.trace.smp(idx).theta_h], obj.N, obj.idx.P_h, [])];
pooled = zeros(obj.N, size(theta, 3), obj.idx.P_c + obj.idx.P_h);
for n = 1:obj.N
    tmp = permute(theta(n,:,:), [1 3 2]);
    pooled(n, :, :) = bsxfun(@minus, tmp, mean(tmp, 2));
end
pooled = reshape(pooled, [], obj.idx.P_c + obj.idx.P_h);
% calculate SVD on empirical covariance of posterior samples
tmp = cov(pooled);

[rotation, scales] = svd(tmp);
scales = sqrt(diag(scales));
% limit smallest proposal step size to 5% of maximum step size
scales(scales < max(scales)*.05) = max(scales)*.05;
if max(scales) > 0
    obj.aux.transform.theta = bsxfun(@times, rotation, scales')';
    tmp = sum(log(scales));
    tmp = exp((obj.aux.logdet.theta - tmp)/(obj.idx.P_c + obj.idx.P_h));
    for n = 1:obj.N
        obj.aux.step.theta(n) = obj.aux.step.theta(n)*tmp;
    end 
    obj.aux.logdet.theta = sum(log(scales));  
end

end


%% potential scale reduction factor
function [ psrf ] = mh_psrf( trace, nChains )

nIt = length(trace);
psrf = struct();
for parameter = fieldnames(trace)'
    smpSize = [size(trace(1).(parameter{1})), nIt];
    try
        tmp = permute(reshape([trace(:).(parameter{1})], smpSize), [3 1 2]);
            
        psrf.(parameter{1}) = tapas_huge_psrf( tmp, nChains );
    catch
        warning('TAPAS:HUGE:convergence', [ 'Potential scale reduction ' ...
            'factor could not be calculated for %s.'], parameter{1});
        psrf.(parameter{1}) = NaN(smpSize);
    end
end

end


function [ obj ] = mh_init( obj )
% Initialize Markov chain state for Metropolized Gibbs sampling.
% Requires obj.dcm and obj.prior to be intialized.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
% 


% fMRI forward model
obj.options.fncBold = @bold_gen;

%% priors
if isvector(obj.prior.tau_0)
    obj.prior.tau_0 = diag(obj.prior.tau_0);
end
% adapt obj.prior to collapsed HUGE
prior = struct();
prior.alpha_0   = obj.prior.alpha_0;
prior.m_0       = obj.prior.m_0;
prior.T_0       = inv(obj.prior.S_0).*obj.prior.tau_0;
prior.s_0       = - log(diag(obj.prior.S_0)');
prior.nu_0 = obj.prior.nu_0;
if isscalar(prior.nu_0)
    prior.nu_0  = repmat(prior.nu_0, obj.idx.P_c, 1);
end
prior.Pi_h      = inv(obj.prior.Sigma_h);
prior.mu_h      = obj.prior.mu_h;
% lambda: BOLD-to-Noise ratio in log-space 
prior.lambda_0  = repmat(log(obj.prior.mu_lambda), 1, obj.R);
prior.omega_0   = repmat(4/(log(obj.prior.s2_lambda).^2), obj.R, 1);

obj.prior = prior;

%% starting values
init.pi = ones(1, obj.K)./obj.K;
init.mu = obj.options.start.clusters;
init.kappa = repmat(obj.prior.s_0, obj.K, 1);
init.theta_c = obj.options.start.subjects(:, 1:obj.idx.P_c);
init.theta_h = obj.options.start.subjects(:, obj.idx.P_c+1:end);
init.lambda = repmat(obj.prior.lambda_0, obj.N, 1);
% randomizing starting values
if obj.options.nvp.randomize
    tmp = obj.options.start.gmm;
    if obj.options.mh.nSteps.weights
        init.pi = init.pi + exp(randn(size(init.pi))*tmp); %%% TODO draw from prior 
        init.pi = init.pi./sum(init.pi); %%% requires samples from dirichlet 
    end
    if obj.options.mh.nSteps.clusters
        init.kappa = init.kappa + randn(size(init.kappa))*tmp;
    end
    tmp = obj.options.start.dcm;
    if obj.options.mh.nSteps.dcm
        init.lambda = init.lambda + randn(size(init.lambda))*tmp;
    end
end

%% switch to single precision
if obj.options.bSinglePrec
    init.pi = single(init.pi);
    init.mu = single(init.mu);
    init.kappa = single(init.kappa);
    init.theta_c = single(init.theta_c);
    init.theta_h = single(init.theta_h);
    init.lambda = single(init.lambda);
end

%% initialize state of Markov chain
obj.aux.l2pi = -.5*obj.idx.P_c*log(2*pi);
% step size
obj.aux.step = obj.options.mh.stepSize;
obj.aux.step.mu = repmat(obj.aux.step.mu, obj.K, 1);
% obj.aux.step.mu = ones(obj.K, 1);
obj.aux.step.kappa = repmat(obj.aux.step.kappa, obj.K, 1);
obj.aux.step.theta = repmat(obj.aux.step.theta, obj.N, 1);
% obj.aux.step.theta = ones(obj.N, 1);
obj.aux.step.lambda = repmat(obj.aux.step.lambda, obj.N, 1);

% transforming proposal distribution
obj.aux.transform = struct();
obj.aux.transform.mu = repmat(eye(obj.idx.P_c), 1, 1, obj.K);
obj.aux.logdet.mu = zeros(obj.K, 1);
% obj.aux.transform.theta = repmat(eye(obj.idx.P_c + obj.idx.P_h), 1, 1, obj.N);
obj.aux.transform.theta = eye(obj.idx.P_c + obj.idx.P_h);
% obj.aux.logdet.theta = zeros(obj.N, 1);
obj.aux.logdet.theta = 0;

obj.aux.sample = init; % obj.aux sample
obj.aux.nProp = struct(); % number of proposals
obj.aux.nAccept = struct(); % number of accepted proposals
obj.aux.lpr = struct(); % log obj.prior
obj.aux.lpr.llh = zeros(obj.N,1); % log likelihood

% weights
obj.aux.nProp.pi = 0;
obj.aux.nAccept.pi = 0;
pic = fliplr(cumsum(obj.aux.sample.pi(end:-1:1)));
pis = obj.aux.sample.pi./pic;
pic = pic(2:end-1);
pis = pis(1:end-1);
obj.aux.sample.pi_u = tapas_huge_logit(pis) + log(obj.K-1:-1:1);

obj.aux.lpr.pi = max(log(obj.aux.sample.pi)*(obj.prior.alpha_0 - 1) ...
     + sum(log(pis)) + sum(log(1-pis)) + sum(log(pic)), -realmax);   
% dPi = obj.aux.sample.pi_u - obj.prior.mu_pi; % Gaussian prior in 
% obj.aux.lpr.pi = -dPi'.^2*obj.prior.pi_pi/2; % unconstrained space

% clusters
obj.aux.nProp.mu = 0;
obj.aux.nAccept.mu = zeros(obj.K, 1);
obj.aux.nAccept.mu_rsk = zeros(obj.K, 2);
obj.aux.nProp.kappa = 0;
obj.aux.nAccept.kappa = zeros(obj.K, 1);
obj.aux.rho = zeros(obj.N, obj.K);
obj.aux.lpr.mu = zeros(1, obj.K);
obj.aux.lpr.kappa = zeros(1, obj.K);
for k = 1:obj.K
    dmu_k = obj.aux.sample.mu(k,:) - obj.prior.m_0;
    obj.aux.lpr.mu(k) = -.5*dmu_k*obj.prior.T_0*dmu_k';
    obj.aux.lpr.kappa(k) = -.5*(obj.aux.sample.kappa(k,:) - ...
        obj.prior.s_0).^2*obj.prior.nu_0;
    dtheta_c = bsxfun(@minus, obj.aux.sample.theta_c, obj.aux.sample.mu(k,:));
    obj.aux.rho(:,k) = -.5*dtheta_c.^2*exp(obj.aux.sample.kappa(k,:)') ...
        + .5*sum(obj.aux.sample.kappa(k,:)) + obj.aux.l2pi;
end
obj.aux.rho_max = max(obj.aux.rho, [], 2);
obj.aux.rho = bsxfun(@minus, obj.aux.rho, obj.aux.rho_max);

% DCM parameters
obj.aux.nProp.theta = 0;
obj.aux.nAccept.theta = zeros(obj.N, 1);
obj.aux.lpr.theta_c = log(exp(obj.aux.rho)*obj.aux.sample.pi(:)) ...
         + obj.aux.rho_max;
dtheta_h = bsxfun(@minus, obj.aux.sample.theta_h, obj.prior.mu_h);
obj.aux.lpr.theta_h = -.5*sum((dtheta_h*obj.prior.Pi_h).*dtheta_h, 2);

obj.aux.lvarBold = zeros(obj.N, obj.R); % log-variance of BOLD response
obj.aux.q_r = zeros(obj.N, 1); % number of scans
obj.aux.epsilon = cell(obj.N, 1);
for n = 1:obj.N
    % number of scans
    obj.aux.q_r(n) = size(obj.data(n).bold, 1);
    % log-variance of BOLD response (per subject and region)
    obj.aux.lvarBold(n,:) = max(log(var(obj.data(n).bold)), ...
        obj.const.minLogVar);
    theta = [obj.aux.sample.theta_c(n,:), obj.aux.sample.theta_h(n,:)];
    epsilon = obj.bold_gen( theta, obj.data(n), obj.inputs(n), ...
        obj.options.hemo, obj.R, obj.L, obj.idx );
    assert(~(any(isnan(epsilon(:))) || any(isinf(epsilon(:)))), ...
        'TAPAS:HUGE:Init:Stability',...
        'Starting values for subject %u lead to instable DCM.', n);
    obj.aux.epsilon{n} = epsilon;
end

% Noise precision (hyperparameters)
obj.aux.nProp.lambda = 0;
obj.aux.nAccept.lambda = zeros(obj.N,1);
dLambda = bsxfun(@minus, obj.aux.sample.lambda, obj.prior.lambda_0);
obj.aux.lpr.lambda = -.5*dLambda.^2*obj.prior.omega_0(:);

%%% TODO add support for confounds

% log-likelihood
for n = 1:obj.N
    obj.aux.lpr.llh(n) = ...
        -.5*sum(obj.aux.epsilon{n}.^2*exp(obj.aux.sample.lambda(n,:) ...
            - obj.aux.lvarBold(n,:))') ...
        +.5*obj.aux.q_r(n)*sum(obj.aux.sample.lambda(n,:)...
            - obj.aux.lvarBold(n,:));
end

% special proposals
obj.aux.nProp.sp = [0;0];
obj.aux.nAccept.sp = [0;0];

end


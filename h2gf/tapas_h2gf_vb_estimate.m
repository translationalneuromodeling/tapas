function [posterior] = tapas_h2gf_vb_estimate(r)
%% See tapas_fitModel documentation
%
% Input
%       
% Output
%       

% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Estimate mode of posterior parameter distribution (MAP estimate)
r = optim(r, r.c_prc.prc_fun, r.c_obs.obs_fun, r.c_opt.opt_algo);

% Separate perceptual and observation parameters
n_prcpars = length(r.c_prc.priormus);
ptrans_prc = r.optim.final(1:n_prcpars);
ptrans_obs = r.optim.final(n_prcpars+1:end);

% Transform MAP parameters back to their native space
[dummy, r.p_prc]   = r.c_prc.transp_prc_fun(r, ptrans_prc);
[dummy, r.p_obs]   = r.c_obs.transp_obs_fun(r, ptrans_obs);
r.p_prc.p      = r.c_prc.transp_prc_fun(r, ptrans_prc);
r.p_obs.p      = r.c_obs.transp_obs_fun(r, ptrans_obs);

% Store transformed MAP parameters
r.p_prc.ptrans = ptrans_prc;
r.p_obs.ptrans = ptrans_obs;

% Store representations at MAP estimate
r.traj = r.c_prc.prc_fun(r, r.p_prc.ptrans, 'trans');

% Print results
ftbrm = {'p', 'ptrans'};
dispprc = rmfield(r.p_prc, ftbrm);
dispobs = rmfield(r.p_obs, ftbrm);

disp(' ')
disp('Results:');
disp(' ')
disp('Parameter estimates for the perceptual model:');
disp(dispprc)
if ~isempty(fieldnames(dispobs))
    disp(' ')
    disp('Parameter estimates for the observation model:');
    disp(dispobs)
end
disp('Model quality:');
disp(['    LME (more is better): ' num2str(r.optim.LME)])
disp(['    AIC (less is better): ' num2str(r.optim.AIC)])
disp(['    BIC (less is better): ' num2str(r.optim.BIC)])
disp(' ')
disp(['    AIC and BIC are approximations to -2*LME = ' num2str(-2*r.optim.LME) '.'])
disp(' ')

return;


function r = optim(r, prc_fun, obs_fun, opt_algo)

% Use means of priors as starting values for optimization for optimized parameters (and as values
% for fixed parameters)
init = [r.c_prc.priormus, r.c_obs.priormus];

% Determine indices of parameters to optimize (i.e., those that are not fixed or NaN)
opt_idx = [r.c_prc.priorsas, r.c_obs.priorsas];
opt_idx(isnan(opt_idx)) = 0;
opt_idx = find(opt_idx);

% Number of perceptual and observation parameters
n_prcpars = length(r.c_prc.priormus);
n_obspars = length(r.c_obs.priormus);

% Construct the objective function to be MINIMIZED:
% The negative log-joint as a function of a single parameter vector
nlj = @(p) [negLogJoint(r, prc_fun, obs_fun, p(1:n_prcpars), p(n_prcpars+1:n_prcpars+n_obspars))];

% Check whether priors are in a region where the objective function can be evaluated
[dummy1, dummy2, rval, err] = nlj(init);
if rval ~= 0
    rethrow(err);
end

% The objective function is now the negative log joint restricted
% with respect to the parameters that are not optimized
obj_fun = @(p_opt) restrictfun(nlj, init, opt_idx, p_opt);

% Optimize
disp(' ')
disp('Optimizing...')
r.optim = opt_algo(obj_fun, init(opt_idx)', r.c_opt);

% Replace optimized values in init with arg min values
final = init;
final(opt_idx) = r.optim.argMin';
r.optim.final = final;

% Get the negative log-joint and negative log-likelihood
[negLj, negLl] = nlj(final);

% Calculate the covariance matrix Sigma and the log-model evidence (as approximated
% by the negative variational free energy under the Laplace assumption).
disp(' ')
disp('Calculating the log-model evidence (LME)...')
d     = length(opt_idx);
Sigma = NaN(d);
Corr  = NaN(d);
LME   = NaN;

options.init_h    = 1;
options.min_steps = 10;

% Numerical computation of the Hessian of the negative log-joint at the MAP estimate
H = tapas_riddershessian(obj_fun, r.optim.argMin, options);

% Use the Hessian from the optimization, if available,
% if the numerical Hessian is not positive definite
if any(isinf(H(:))) || any(isnan(H(:))) || any(eig(H)<=0)
    if isfield(r.optim, 'T')
        % Hessian of the negative log-joint at the MAP estimate
        H     = inv(r.optim.T);
        % Parameter covariance 
        Sigma = r.optim.T;
        % Parameter correlation
        Corr = tapas_Cov2Corr(Sigma);
        % Log-model evidence ~ negative variational free energy
        LME   = -r.optim.valMin + 1/2*log(1/det(H)) + d/2*log(2*pi);
    else
        disp('Warning: Cannot calculate Sigma and LME because the Hessian is not positive definite.')
    end
else
    % Parameter covariance
    Sigma = inv(H);
    % Parameter correlation
    Corr = tapas_Cov2Corr(Sigma);
    % Log-model evidence ~ negative variational free energy
    LME = -r.optim.valMin + 1/2*log(1/det(H)) + d/2*log(2*pi);
end

% Calculate accuracy and complexity (LME = accu - comp)
accu = -negLl;
comp = accu -LME;

% Calculate AIC and BIC
if ~isempty(r.y)
    ndp = sum(~isnan(r.y(:,1)));
else
    ndp = sum(~isnan(r.u(:,1)));
end
AIC  = 2*negLl +2*d;
BIC  = 2*negLl +d*log(ndp);

r.optim.H     = H;
r.optim.Sigma = Sigma;
r.optim.Corr  = Corr;
r.optim.negLl = negLl;
r.optim.negLj = negLj;
r.optim.LME   = LME;
r.optim.accu  = accu;
r.optim.comp  = comp;
r.optim.AIC   = AIC;
r.optim.BIC   = BIC;

return;

% --------------------------------------------------------------------------------------------------
function [negLogJoint, negLogLl, rval, err] = negLogJoint(r, prc_fun, obs_fun, ptrans_prc, ptrans_obs)
% Returns the the negative log-joint density for perceptual and observation parameters

% Calculate perceptual trajectories. The columns of the matrix infStates contain the trajectories of
% the inferred states (according to the perceptual model) that the observation model bases its
% predictions on.
try
    [dummy, infStates] = prc_fun(r, ptrans_prc, 'trans');
catch err
    negLogJoint = realmax;
    negLogLl = realmax;
    % Signal that something has gone wrong
    rval = -1;
    return;
end

% Calculate the log-likelihood of observed responses given the perceptual trajectories,
% under the observation model
trialLogLls = obs_fun(r, infStates, ptrans_obs);
logLl = nansum(trialLogLls);
negLogLl = -logLl;

% Calculate the log-prior of the perceptual parameters.
% Only parameters that are neither NaN nor fixed (non-zero prior variance) are relevant.
prc_idx = r.c_prc.priorsas;
prc_idx(isnan(prc_idx)) = 0;
prc_idx = find(prc_idx);

logPrcPriors = -1/2.*log(8*atan(1).*r.c_prc.priorsas(prc_idx)) - 1/2.*(ptrans_prc(prc_idx) - r.c_prc.priormus(prc_idx)).^2./r.c_prc.priorsas(prc_idx);
logPrcPrior  = sum(logPrcPriors);

% Calculate the log-prior of the observation parameters.
% Only parameters that are neither NaN nor fixed (non-zero prior variance) are relevant.
obs_idx = r.c_obs.priorsas;
obs_idx(isnan(obs_idx)) = 0;
obs_idx = find(obs_idx);

logObsPriors = -1/2.*log(8*atan(1).*r.c_obs.priorsas(obs_idx)) - 1/2.*(ptrans_obs(obs_idx) - r.c_obs.priormus(obs_idx)).^2./r.c_obs.priorsas(obs_idx);
logObsPrior  = sum(logObsPriors);

negLogJoint = -(logLl + logPrcPrior + logObsPrior);

% Signal that all has gone right
err = [];
rval = 0;

return;

% --------------------------------------------------------------------------------------------------
function val = restrictfun(f, arg, free_idx, free_arg)
% This is a helper function for the construction of file handles to
% restricted functions.
%
% It returns the value of a function restricted to subset of the
% arguments of the input function handle. The input handle takes
% *one* vector as its argument.
% 
% INPUT:
%   f            The input function handle
%   arg          The argument vector for the input function containing the
%                fixed values of the restricted arguments (plus dummy values
%                for the free arguments)
%   free_idx     The index numbers of the arguments that are not restricted
%   free_arg     The values of the free arguments

% Replace the dummy arguments in arg
arg(free_idx) = free_arg;

% Evaluate
val = f(arg);

return;

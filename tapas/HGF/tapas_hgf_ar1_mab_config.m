function c = tapas_hgf_ar1_mab_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Hierarchical Gaussian Filter (HGF) for AR(1)
% processes in multi-armed bandit situations. The template for such
% a situation is the task from
%
% Daw ND, O’Doherty JP, Dayan P, Seymour B, and Dolan RJ. (2006).
% Cortical substrates for exploratory decisions in humans. Nature, 441(7095), 876–879.
%
% The HGF is the model introduced in 
%
% Mathys C, Daunizeau J, Friston KJ, and Stephan KE. (2011). A Bayesian foundation
% for individual learning under uncertainty. Frontiers in Human Neuroscience, 5:39.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The HGF configuration consists of the priors of parameters and initial values. All priors are
% Gaussian in the space where the quantity they refer to is estimated. They are specified by their
% sufficient statistics: mean and variance (NOT standard deviation).
% 
% Quantities are estimated in their native space if they are unbounded (e.g., the omegas). They are
% estimated in log-space if they have a natural lower bound at zero (e.g., the sigmas).
% 
% The phis, kappas, and theta) are estimated in 'logit-space' because bounding them above (in
% addition to their natural lower bound at zero) is an effective means of preventing the exploration
% of parameter regions where the assumptions underlying the variational inversion (cf. Mathys et
% al., 2011) no longer hold.
% 
% 'Logit-space' is a logistic sigmoid transformation of native space with a variable upper bound
% a>0:
% 
% tapas_logit(x) = ln(x/(a-x)); x = a/(1+exp(-tapas_logit(x)))
%
% Parameters can be fixed (i.e., set to a fixed value) by setting the variance of their prior to
% zero. Aside from being useful for model comparison, the need for this arises whenever the scale
% and origin at the j-th level are arbitrary. This is the case if the observation model does not
% contain the representations mu_j and sigma_j. A choice of scale and origin is then implied by
% fixing the initial value mu_j_0 of mu_j and either kappa_j-1 or omega_j-1.
%
% The kappas and theta can be fixed to an arbitrary value by setting the upper bound to twice that
% value and the mean as well as the variance of the prior to zero (this follows immediately from
% the logit transform above).
% 
% Fitted trajectories can be plotted by using the command
%
% >> tapas_hgf_ar1_plotTraj(est)
% 
% where est is the stucture returned by tapas_fitModel. This structure contains the estimated
% perceptual parameters in est.p_prc and the estimated trajectories of the agent's
% representations (cf. Mathys et al., 2011). Their meanings are:
%              
%         est.p_prc.mu_0       row vector of initial values of mu (in ascending order of levels)
%         est.p_prc.sa_0       row vector of initial values of sigma (in ascending order of levels)
%         est.p_prc.phi        row vector of phis
%         est.p_prc.m          row vector of ms
%         est.p_prc.ka         row vector of kappas (in ascending order of levels)
%         est.p_prc.om         row vector of omegas (in ascending order of levels)
%         est.p_prc.th         theta
%         est.p_prc.al         alpha
%
%         est.traj.mu          mu (rows: trials, columns: levels)
%         est.traj.sa          sigma (rows: trials, columns: levels)
%         est.traj.muhat       prediction of mu (rows: trials, columns: levels)
%         est.traj.sahat       precisions of predictions (rows: trials, columns: levels)
%         est.traj.w           weighting factors (rows: trials, columns: levels)
%         est.traj.da          volatility prediction errors  (rows: trials, columns: levels)
%         est.traj.dau         input prediction error
%
% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = tapas_fitModel([], u, 'tapas_hgf_ar1_config', 'tapas_bayes_optimal_config');
%
%   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
%   this file here, so choose them wide and loose to let the inputs influence the result). You can
%   then use the optimal parameters as your new prior means for the perceptual parameters.
%
% - If you get an error saying that the prior means are in a region where model assumptions are
%   violated, or if you simply get implausible trajectories (e.g., huge jumps in any of the mus), try
%   lowering the upper bound on theta.
%
% - Alternatives are lowering the upper bounds on the kappas, if they are not fixed, or adjusting
%   the values of the kappas or omegas, if any of them are fixed.
%
% - If the negative free energy F cannot be calculated because the Hessian poses problems, look at
%   est.optim.H and fix the parameters that lead to NaNs.
%
% - Your guide to all these adjustments is the negative free energy F. Whenever F increases by at
%   least 3 across datasets, the adjustment was a good idea and can be justified by just this: F
%   increased, so you had a better model.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Model name
c.model = 'tapas_hgf_ar1_mab';

% Number of bandits
c.n_bandits = 3;

% Number of levels (minimum: 2)
c.n_levels = 2;

% Input intervals
% If input intervals are irregular, the last column of the input
% matrix u has to contain the interval between inputs k-1 and k
% in the k-th row, and this flag has to be set to true
c.irregular_intervals = false;

% Sufficient statistics of Gaussian parameter priors

% PLACEHOLDER VALUES
% It is often convenient to set some priors to values
% derived from the inputs. This can be achieved by
% using placeholder values. The available placeholders
% are:
%
% 99991   Value of the first input
%         Usually a good choice for mu_0mu(1)
% 99992   Variance of the first 20 inputs
%         Usually a good choice for mu_0sa(1)
% 99993   Log-variance of the first 20 inputs
%         Usually a good choice for logsa_0mu(1)
%         and logalmu
% 99994   Log-variance of the first 20 inputs minus two
%         Usually a good choice for ommu(1)

% Initial mus and sigmas
% Format: row vectors of length n_levels
% For all but the first level, this is usually best
% kept fixed to 1 (determines origin on x_i-scale).
c.mu_0mu = [50, 1];
c.mu_0sa = [0, 0];

c.logsa_0mu = [log(70), log(0.1)];
c.logsa_0sa = [      0,        0];

% Phis
% Format: row vector of length n_levels.
% Phi is estimated in logit-space because it is
% bounded between 0 and 1
% Fix this to zero (leading to a Gaussian random walk) by
% setting logitphimu = -Inf; logitphisa = 0;
c.logitphimu = [tapas_logit(0.02,1), -Inf];
c.logitphisa = [            1,    0];

% ms
% Format: row vector of length n_levels.
% This should be fixed for all levels where the omega of
% the next lowest level is not fixed because that offers
% an alternative parametrization of the same model.
c.mmu = [ 50, c.mu_0mu(2)];
c.msa = [8^2,           0];

% Upper bounds on kappas (lower bound is always zero)
% Format: row vector of length n_levels-1
c.kaub = [2];

% Kappas
% Format: row vector of length n_levels-1.
% This should be fixed (preferably to 1) if the observation model
% does not use mu_i+1 (kappa then determines the scaling of x_i+1).
c.logitkamu = [0];
c.logitkasa = [0];

% Omegas
% Format: row vector of length n_levels-1
c.ommu = [  4];
c.omsa = [4^2];

% Upper bound on theta (lower bound is always zero)
% Format: scalar
% NOTE: If set to zero, this will be automatically
% adjusted to the highest value (not greater than 2)
% for which the assumptions underlying the variational
% inversion of the HGF still hold.
c.thub = 1;

% Theta
% Format: scalar
c.logitthmu = 0;
c.logitthsa = 10^2;

% Alpha
% Format: scalar
% Fix this to zero (no percpeptual uncertainty) by setting
% logalmu = -Inf; logalsa = 0;
c.logalmu = log(128);
c.logalsa = 0;

% Gather prior settings in vectors
c.priormus = [
    c.mu_0mu,...
    c.logsa_0mu,...
    c.logitphimu,...
    c.mmu,...
    c.logitkamu,...
    c.ommu,...
    c.logitthmu,...
    c.logalmu,...
         ];

c.priorsas = [
    c.mu_0sa,...
    c.logsa_0sa,...
    c.logitphisa,...
    c.msa,...
    c.logitkasa,...
    c.omsa,...
    c.logitthsa,...
    c.logalsa,...
         ];

% Check whether we have the right number of priors
expectedLength = 4*c.n_levels+2*(c.n_levels-1)+2;
if length([c.priormus, c.priorsas]) ~= 2*expectedLength;
    error('Prior definition does not match number of levels.')
end

% Model function handle
c.prc_fun = @tapas_hgf_ar1_mab;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_hgf_ar1_mab_transp;

return;

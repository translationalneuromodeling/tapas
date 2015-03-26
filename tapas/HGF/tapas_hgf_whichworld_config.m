function c = tapas_hgf_whichworld_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Hierarchical Gaussian Filter (HGF) for binary inputs restricted
% to 3 levels, no drift, and no inputs at irregular intervals, in the absence of perceptual
% uncertainty.
%
% This model deals with the situation where an agent has to determine in which of several
% possible worlds, each characterized by a different Bernoulli distribution of binary outcomes,
% he is currently living in. The probabilities of the different possible distributions are
% assumed to perform Gaussian random walks in logit space. The volatilities of all of these walks
% are determined by the same higher-level state x_3 in standard HGF fashion.
%
% The HGF is the model introduced in 
%
% Mathys C, Daunizeau J, Friston, KJ, and Stephan KE. (2011). A Bayesian foundation
% for individual learning under uncertainty. Frontiers in Human Neuroscience, 5:39.
%
% This file refers to BINARY inputs (Eqs 1-3 in Mathys et al., (2011));
% for continuous inputs, refer to hgf_config.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The HGF configuration consists of the priors of parameters and initial values. All priors are
% Gaussian in the space where the quantity they refer to is estimated. They are specified by their
% sufficient statistics: mean and variance (NOT standard deviation).
% 
% Quantities are estimated in their native space if they are unbounded (e.g., omega). They are
% estimated in log-space if they have a natural lower bound at zero (e.g., sigma2).
% 
% Kappa and theta are estimated in 'logit-space' because bounding them above (in addition to
% their natural lower bound at zero) is an effective means of preventing the exploration of
% parameter regions where the assumptions underlying the variational inversion (cf. Mathys et
% al., 2011) no longer hold.
% 
% 'Logit-space' is a logistic sigmoid transformation of native space with a variable upper bound
% a>0:
% 
% logit(x) = ln(x/(a-x)); x = a/(1+exp(-logit(x)))
%
% Parameters can be fixed (i.e., set to a fixed value) by setting the variance of their prior to
% zero. Aside from being useful for model comparison, the need for this arises whenever the scale
% and origin of x3 are arbitrary. This is the case if the observation model does not contain the
% representations mu3 and sigma3 from the third level. A choice of scale and origin is then
% implied by fixing the initial value mu3_0 of mu3 and either kappa or omega.
%
% Kappa and theta can be fixed to an arbitrary value by setting the upper bound to twice that
% value and the mean as well as the variance of the prior to zero (this follows immediately from
% the logit transform above).
% 
% Fitted trajectories can be plotted by using the command
%
% >> tapas_hgf_whichworld_plotTraj(est)
% 
% where est is the stucture returned by fitModel. This structure contains the estimated
% perceptual parameters in est.p_prc and the estimated trajectories of the agent's
% representations (cf. Mathys et al., 2011). Their meanings are:
%              
%         est.p_prc.mu2_0      initial values of the mu2s
%         est.p_prc.sa2_0      initial values of the sigma2s
%         est.p_prc.mu3_0      initial value of mu3
%         est.p_prc.sa3_0      initial value of sigma3
%         est.p_prc.ka         kappa
%         est.p_prc.om         omega
%         est.p_prc.th         theta
%
%         est.traj.mu          mu
%         est.traj.sa          sigma
%         est.traj.muhat       prediction mean
%         est.traj.sahat       prediction variance
%         est.traj.v           inferred variances of random walks
%         est.traj.w           weighting factor of informational and environmental uncertainty at the 2nd level
%         est.traj.da          prediction errors
%         est.traj.ud          updates with respect to prediction
%         est.traj.psi         precision weights on prediction errors
%         est.traj.epsi        precision-weighted prediction errors
%         est.traj.wt          full weights on prediction errors (at the first level,
%                                  this is the learning rate)
%
% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = tapas_fitModel([], u, 'tapas_hgf_whichworld_config', 'tapas_bayes_optimal_whichworld_config');
%
%   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
%   this file here, so choose them wide and loose to let the inputs influence the result). You can
%   then use the optimal parameters as your new prior means for the perceptual parameters.
%
% - If you get an error saying that the prior means are in a region where model assumptions are
%   violated, lower the prior means of the omegas, starting with the highest level and proceeding
%   downwards.
%
% - Alternatives are lowering the prior mean of kappa, if they are not fixed, or adjusting
%   the values of the kappas or omegas, if any of them are fixed.
%
% - If the log-model evidence cannot be calculated because the Hessian poses problems, look at
%   est.optim.H and fix the parameters that lead to NaNs.
%
% - Your guide to all these adjustments is the log-model evidence (LME). Whenever the LME increases
%   by at least 3 across datasets, the adjustment was a good idea and can be justified by just this:
%   the LME increased, so you had a better model.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013-2014 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Model name
c.model = 'hgf_whichworld';

% Number of worlds
c.nw = 2;

% Upper bound for kappa and theta (lower bound is always zero)
c.kaub = 2;
c.thub = 2;

% Sufficient statistics of Gaussian parameter priors

% Initial mu2
c.mu2_0mu = [tapas_logit(1/2,1), tapas_logit(1/2,1)];
c.mu2_0sa = [           0,            0];

% Initial sigma2
c.logsa2_0mu = [log(1), log(1)];
c.logsa2_0sa = [     1,      1];

% Initial mu3
% Usually best kept fixed to 1 (determines origin on x3-scale).
c.mu3_0mu = 1;
c.mu3_0sa = 0;

% Initial sigma3
c.logsa3_0mu = log(0.1);
c.logsa3_0sa = 1;

% Kappa
% This should be fixed (preferably to 1) if the observation model
% does not use mu3 (kappa then determines the scaling of x3).
c.logitkamu = 0;
c.logitkasa = 0;

% Omega
c.ommu = 0;
c.omsa = 5^2;

% Theta
c.logitthmu = 0;
c.logitthsa = 2;

% m
c.mmu = 0;
c.msa = 0;

% Phi
c.logitphimu = tapas_logit(0.1,1);
c.logitphisa = 2;

% Gather prior settings in vectors
c.priormus = [
    c.mu2_0mu,...
    c.logsa2_0mu,...
    c.mu3_0mu,...
    c.logsa3_0mu,...
    c.logitkamu,...
    c.ommu,...
    c.logitthmu,...
    c.mmu,...
    c.logitphimu,...
         ];

c.priorsas = [
    c.mu2_0sa,...
    c.logsa2_0sa,...
    c.mu3_0sa,...
    c.logsa3_0sa,...
    c.logitkasa,...
    c.omsa,...
    c.logitthsa,...
    c.msa,...
    c.logitphisa,...
         ];

% Model function handle
c.prc_fun = @tapas_hgf_whichworld;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_hgf_whichworld_transp;

return;

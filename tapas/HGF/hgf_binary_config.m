function c = hgf_binary_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Hierarchical Gaussian Filter (HGF)
% for binary inputs in the absence of perceptual uncertainty.
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
% Quantities are estimated in their native space if they are unbounded (e.g., the omegas). They are
% estimated in log-space if they have a natural lower bound at zero (e.g., the sigmas).
% 
% The kappas (and theta) are estimated in 'logit-space' because bounding them above (in addition to
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
% >> hgf_binary_plotTraj(est)
% 
% where est is the stucture returned by fitModel. This structure contains the estimated
% perceptual parameters in est.p_prc and the estimated trajectories of the agent's
% representations (cf. Mathys et al., 2011). Their meanings are:
%              
%         est.p_prc.mu_0       row vector of initial values of mu (in ascending order of levels)
%         est.p_prc.sa_0       row vector of initial values of sigma (in ascending order of levels)
%         est.p_prc.rho        row vector of rhos (representing drift; in ascending order of levels)
%         est.p_prc.ka         row vector of kappas (in ascending order of levels)
%         est.p_prc.om         row vector of omegas (in ascending order of levels)
%         est.p_prc.th         theta
%
% Note that the first entry in all of the row vectors will be NaN because, at the first level,
% these parameters are either determined by the second level (mu_0 and sa_0) or undefined (rho,
% kappa, and omega).
%
%         est.traj.mu          mu (rows: trials, columns: levels)
%         est.traj.sa          sigma (rows: trials, columns: levels)
%         est.traj.muhat       prediction of mu (rows: trials, columns: levels)
%         est.traj.sahat       precisions of predictions (rows: trials, columns: levels)
%         est.traj.w           weighting factors (rows: trials, columns: levels)
%         est.traj.da          volatility prediction errors  (rows: trials, columns: levels)
%         est.traj.dau         input prediction error
%
% Note that in the absence of sensory uncertainty (which is the assumption here), the first
% column of mu, corresponding to the first level, will be equal to the inputs. Likewise, the
% first column of sa will be 0 always.
%
% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = fitModel([], u, 'hgf_binary_config', 'bayes_optimal_binary_config');
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
c.model = 'hgf_binary';

% Number of levels (minimum: 3)
c.n_levels = 3;

% Input intervals
% If input intervals are irregular, the last column of the input
% matrix u has to contain the interval between inputs k-1 and k
% in the k-th row, and this flag has to be set to true
c.irregular_intervals = false;

% Sufficient statistics of Gaussian parameter priors

% Initial mus and sigmas
% Format: row vectors of length n_levels
% For all but the first two levels, this is usually best
% kept fixed to 1 (determines origin on x_i-scale). The 
% first level is NaN because it is determined by the second,
% and the second implies neutrality between outcomes when it
% is centered at 0.
c.mu_0mu = [NaN, 0, 1];
c.mu_0sa = [NaN, 0, 0];

c.logsa_0mu = [NaN, log(1), log(1)];
c.logsa_0sa = [NaN,      1,      1];

% Rhos
% Format: row vector of length n_levels.
% Undefined (therefore NaN) at the first level.
% Fix this to zero to turn off drift.
c.rhomu = [NaN, 0, 0];
c.rhosa = [NaN, 0, 0];

% Upper bounds on kappas (lower bound is always zero).
% Format: row vector of length n_levels-1.
% Undefined (therefore NaN) at the first level.
c.kaub = [NaN, 2];

% Kappas
% Format: row vector of length n_levels-1.
% Undefined (therefore NaN) at the first level.
% This should be fixed (preferably to 1) if the observation model
% does not use mu_i+1 (kappa then determines the scaling of x_i+1).
c.logitkamu = [NaN, 0];
c.logitkasa = [NaN, 0];

% Omegas
% Format: row vector of length n_levels-1.
% Undefined (therefore NaN) at the first level.
c.ommu = [NaN,   -2];
c.omsa = [NaN, 4^2];

% Upper bound on theta (lower bound is always zero).
% Format: scalar
c.thub = 0.1;

% Theta
% Format: scalar
c.logitthmu = 0;
c.logitthsa = 4^2;

% Gather prior settings in vectors
c.priormus = [
    c.mu_0mu,...
    c.logsa_0mu,...
    c.rhomu,...
    c.logitkamu,...
    c.ommu,...
    c.logitthmu,...
         ];

c.priorsas = [
    c.mu_0sa,...
    c.logsa_0sa,...
    c.rhosa,...
    c.logitkasa,...
    c.omsa,...
    c.logitthsa,...
         ];

% Check whether we have the right number of priors
expectedLength = 3*c.n_levels+2*(c.n_levels-1)+1;
if length([c.priormus, c.priorsas]) ~= 2*expectedLength;
    error('Prior definition does not match number of levels.')
end

% Model function handle
c.prc_fun = @hgf_binary;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @hgf_binary_transp;

return;

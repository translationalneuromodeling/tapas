function c = tapas_hgf_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Hierarchical Gaussian Filter (HGF)
% for continuous inputs.
%
% The HGF is the model introduced in 
%
% Mathys C, Daunizeau J, Friston, KJ, and Stephan KE. (2011). A Bayesian foundation
% for individual learning under uncertainty. Frontiers in Human Neuroscience, 5:39.
%
% This file refers to CONTINUOUS inputs (Eqs 48ff in Mathys et al., (2011));
% for binary inputs, refer to tapas_hgf_binary_config.
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
% Parameters can be fixed (i.e., set to a fixed value) by setting the variance of their prior to
% zero. Aside from being useful for model comparison, the need for this arises whenever the scale
% and origin at the j-th level are arbitrary. This is the case if the observation model does not
% contain the representations mu_j and sigma_j. A choice of scale and origin is then implied by
% fixing the initial value mu_j_0 of mu_j and either kappa_j-1 or omega_j-1.
%
% Fitted trajectories can be plotted by using the command
%
% >> tapas_hgf_plotTraj(est)
% 
% where est is the stucture returned by tapas_fitModel. This structure contains the estimated
% perceptual parameters in est.p_prc and the estimated trajectories of the agent's
% representations (cf. Mathys et al., 2011). Their meanings are:
%              
%         est.p_prc.mu_0       row vector of initial values of mu (in ascending order of levels)
%         est.p_prc.sa_0       row vector of initial values of sigma (in ascending order of levels)
%         est.p_prc.rho        row vector of rhos (representing drift; in ascending order of levels)
%         est.p_prc.ka         row vector of kappas (in ascending order of levels)
%         est.p_prc.om         row vector of omegas (in ascending order of levels)
%         est.p_prc.pi_u       pi_u (input precision = 1/alpha)
%
%         est.traj.mu          mu (rows: trials, columns: levels)
%         est.traj.sa          sigma (rows: trials, columns: levels)
%         est.traj.muhat       prediction of mu (rows: trials, columns: levels)
%         est.traj.sahat       precisions of predictions (rows: trials, columns: levels)
%         est.traj.v           inferred variance of random walk (rows: trials, columns: levels)
%         est.traj.w           weighting factors (rows: trials, columns: levels)
%         est.traj.da          volatility prediction errors  (rows: trials, columns: levels)
%         est.traj.dau         input prediction error
%         est.traj.ud          updates with respect to prediction  (rows: trials, columns: levels)
%         est.traj.psi         precision weights on prediction errors  (rows: trials, columns: levels)
%         est.traj.epsi        precision-weighted prediction errors  (rows: trials, columns: levels)
%         est.traj.wt          full weights on prediction errors (at the first level,
%                                  this is the learning rate) (rows: trials, columns: levels)
%
% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = tapas_fitModel([], u, 'tapas_hgf_config', 'tapas_bayes_optimal_config');
%
%   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
%   this file here, so choose them wide and loose to let the inputs influence the result). You can
%   then use the optimal parameters as your new prior means for the perceptual parameters.
%
% - If you get an error saying that the prior means are in a region where model assumptions are
%   violated, lower the prior means of the omegas, starting with the highest level and proceeding
%   downwards.
%
% - Alternatives are lowering the prior means of the kappas, if they are not fixed, or adjusting
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
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Model name
c.model = 'hgf';

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
%         Usually a good choice for logsa_0mu(1), and
%         its negative, ie the log-precision of the
%         first 20 inputs, for logpiumu
% 99994   Log-variance of the first 20 inputs minus two
%         Usually a good choice for ommu(1)

% Initial mus and sigmas
% Format: row vectors of length n_levels
% For all but the first level, this is usually best
% kept fixed to 1 (determines origin on x_i-scale).
c.mu_0mu = [99991, 1];
c.mu_0sa = [99992, 0];

c.logsa_0mu = [99993, log(0.1)];
c.logsa_0sa = [    1,        1];

% Rhos
% Format: row vector of length n_levels
% Fix this to zero to turn off drift
c.rhomu = [0, 0];
c.rhosa = [0, 0];

% Kappas
% Format: row vector of length n_levels-1
% This should be fixed (preferably to 1) if the observation model
% does not use mu_i+1 (kappa then determines the scaling of x_i+1).
c.logkamu = [log(1)];
c.logkasa = [     0];

% Omegas
% Format: row vector of length n_levels
c.ommu = [99993,  -4];
c.omsa = [  4^2, 4^2];

% Pi_u
% Format: scalar
% Fix this to zero (no percpeptual uncertainty) by setting
% logpiumu = -Inf; logpiusa = 0;
c.logpiumu = -99993;
c.logpiusa = 2^2;

% Gather prior settings in vectors
c.priormus = [
    c.mu_0mu,...
    c.logsa_0mu,...
    c.rhomu,...
    c.logkamu,...
    c.ommu,...
    c.logpiumu,...
         ];

c.priorsas = [
    c.mu_0sa,...
    c.logsa_0sa,...
    c.rhosa,...
    c.logkasa,...
    c.omsa,...
    c.logpiusa,...
         ];

% Check whether we have the right number of priors
expectedLength = 3*c.n_levels+2*(c.n_levels-1)+2;
if length([c.priormus, c.priorsas]) ~= 2*expectedLength;
    error('tapas:hgf:PriorDefNotMatchingLevels', 'Prior definition does not match number of levels.')
end

% Model function handle
c.prc_fun = @tapas_hgf;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_hgf_transp;

return;

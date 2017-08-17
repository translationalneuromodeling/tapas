function c = tapas_hgf_binary_pu_tbt_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Hierarchical Gaussian Filter (HGF) for binary inputs in the
% *presence* of perceptual uncertainty with the additional twist that perceptual uncertainty changes
% from trial to trial (tbt: trial-by-trial). The uncertainty (alpha) is expected by the model in
% the second column of the input matrix.
%
% The HGF is the model introduced in 
%
% Mathys C, Daunizeau J, Friston, KJ, and Stephan KE. (2011). A Bayesian foundation
% for individual learning under uncertainty. Frontiers in Human Neuroscience, 5:39.
%
% The binary HGF model has since been augmented with a positive factor kappa1 which
% scales the second level with respect to the first, i.e., the relation between the
% first and second level is
%
% p(x1=1|x2) = s(kappa1*x2), where s(.) is the logistic sigmoid.
%
% By default, kappa1 is fixed to 1, leading to the model introduced in Mathys et al. (2011).
%
% This file refers to BINARY inputs (Eqs 1-3 in Mathys et al., (2011));
% for continuous inputs, refer to tapas_hgf_config.
%
% This file refers to UNCERTAIN inputs (Eqs 45-47 in Mathys et al., (2011));
% for inputs without uncertainty, refer to tapas_hgf_binary_config.
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
% >> tapas_hgf_binary_plotTraj(est)
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
%         est.p_prc.al         scalar alpha (perceptual uncertainty)
%         est.p_prc.eta0       scalar eta0 (mean of first input category)
%         est.p_prc.eta1       scalar eta1 (mean of second input category)
%
% Note that the first entry in all of the row vectors will be NaN because, at the first level,
% these parameters are either determined by the second level (mu_0 and sa_0) or undefined (rho,
% kappa, and omega).
%
%         est.traj.mu          mu (rows: trials, columns: levels)
%         est.traj.sa          sigma (rows: trials, columns: levels)
%         est.traj.muhat       prediction of mu (rows: trials, columns: levels)
%         est.traj.sahat       precisions of predictions (rows: trials, columns: levels)
%         est.traj.v           inferred variance of random walk (rows: trials, columns: levels)
%         est.traj.w           weighting factors (rows: trials, columns: levels)
%         est.traj.da          volatility prediction errors  (rows: trials, columns: levels)
%         est.traj.ud          updates with respect to prediction  (rows: trials, columns: levels)
%         est.traj.psi         precision weights on prediction errors  (rows: trials, columns: levels)
%         est.traj.epsi        precision-weighted prediction errors  (rows: trials, columns: levels)
%         est.traj.wt          full weights on prediction errors (at the first level,
%                                  this is the learning rate) (rows: trials, columns: levels)
%
% Note that in the absence of sensory uncertainty (which is the assumption here), the first
% column of mu, corresponding to the first level, will be equal to the inputs. Likewise, the
% first column of sa will be 0 always.
%
% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = tapas_fitModel([], u, 'tapas_hgf_binary_pu_tbt_config', 'tapas_bayes_optimal_binary_config');
%
%   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
%   this file here, so choose them wide and loose to let the inputs influence the result). You can
%   then use the optimal parameters as your new prior means for the perceptual parameters.
%
% - When analyzing a new dataset, take your inputs u and use
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
% Copyright (C) 2016-2017 Rebecca Lawson, Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Model name
c.model = 'hgf_binary_pu_tbt';

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

c.logsa_0mu = [NaN,   log(0.1), log(1)];
c.logsa_0sa = [NaN,          0,      0];

% Rhos
% Format: row vector of length n_levels.
% Undefined (therefore NaN) at the first level.
% Fix this to zero to turn off drift.
c.rhomu = [NaN, 0, 0];
c.rhosa = [NaN, 0, 0];

% Kappas
% Format: row vector of length n_levels-1.
% Fixing log(kappa1) to log(1) leads to the original HGF model.
% Higher log(kappas) should be fixed (preferably to log(1)) if the
% observation model does not use mu_i+1 (kappa then determines the
% scaling of x_i+1).
c.logkamu = [log(1), log(1)];
c.logkasa = [     0,      0];

% Omegas
% Format: row vector of length n_levels.
% Undefined (therefore NaN) at the first level.
c.ommu = [NaN,  -5,  -6];
c.omsa = [NaN, 4^2, 4^2];

% Eta0
% Format: scalar.
c.eta0mu = 0;
c.eta0sa = 0;

% Eta1
% Format: scalar.
c.eta1mu = 1;
c.eta1sa = 0;

% Gather prior settings in vectors
c.priormus = [
    c.mu_0mu,...
    c.logsa_0mu,...
    c.rhomu,...
    c.logkamu,...
    c.ommu,...
    c.eta0mu,...
    c.eta1mu,...
         ];

c.priorsas = [
    c.mu_0sa,...
    c.logsa_0sa,...
    c.rhosa,...
    c.logkasa,...
    c.omsa,...
    c.eta0sa,...
    c.eta1sa,...
         ];
     

 % Check whether we have the right number of priors
expectedLength = 3*c.n_levels+2*(c.n_levels-1)+3;
if length([c.priormus, c.priorsas]) ~= 2*expectedLength; 
    error('tapas:hgf:PriorDefNotMatchingLevels', 'Prior definition does not match number of levels.')
end

% Model function handle
c.prc_fun = @tapas_hgf_binary_pu_tbt;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_hgf_binary_pu_tbt_transp;

return;

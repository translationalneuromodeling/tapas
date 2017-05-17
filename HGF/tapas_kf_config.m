function c = tapas_kf_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Kalman filter
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The Kalman filter configuration consists of the priors of parameters and initial values. All
% priors are Gaussian in the space where the quantity they refer to is estimated. They are specified
% by their sufficient statistics: mean and variance (NOT standard deviation).
% 
% Quantities are estimated in their native space if they are unbounded (e.g., mu_0). They are
% estimated in log-space if they have a natural lower bound at zero (e.g., the pi_u).
% 
% Parameters can be fixed (i.e., set to a fixed value) by setting the variance of their prior to
% zero.
%
% Fitted trajectories can be plotted by using the command
%
% >> tapas_kf_plotTraj(est)
% 
% where est is the stucture returned by tapas_fitModel. This structure contains the estimated
% perceptual parameters in est.p_prc and the estimated trajectories of the filter's
% representations. Their meanings are:
%              
%         est.p_prc.g_0        initial value of gain
%         est.p_prc.mu_0       initial values of hidden state mean
%         est.p_prc.om         process variance
%         est.p_prc.pi_u       observation precision
%
%         est.traj.da          prediction error
%         est.traj.g           gain
%         est.traj.mu          hidden state mean
%
% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = tapas_fitModel([], u, 'tapas_kf_config', 'tapas_bayes_optimal_config');
%
%   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
%   this file here, so choose them wide and loose to let the inputs influence the result). You can
%   then use the optimal parameters as your new prior means for the perceptual parameters.
%
% - If the log-model evidence cannot be calculated because the Hessian poses problems, look at
%   est.optim.H and fix the parameters that lead to NaNs.
%
% - Your guide to all these adjustments is the log-model evidence (LME). Whenever the LME increases
%   by at least 3 across datasets, the adjustment was a good idea and can be justified by just this:
%   the LME increased, so you had a better model.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Model name
c.model = 'Kalman filter';

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

% Initial gain
c.logg_0mu = 0.1;
c.logg_0sa = 1;

% Initial hidden state mean
c.mu_0mu = 99991;
c.mu_0sa = 99992;

% Process variance
c.ommu = 99993;
c.omsa = 1;

% Pi_u
% Fix this to zero (no percpeptual uncertainty) by setting
% logpiumu = -Inf; logpiusa = 0;
c.logpiumu = -99993;
c.logpiusa = 1;

% Gather prior settings in vectors
c.priormus = [
    c.logg_0mu,...
    c.mu_0mu,...
    c.ommu,...
    c.logpiumu,...
         ];

c.priorsas = [
    c.logg_0sa,...
    c.mu_0sa,...
    c.omsa,...
    c.logpiusa,...
         ];

% Model function handle
c.prc_fun = @tapas_kf;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_kf_transp;

return;

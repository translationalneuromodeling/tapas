function c = tapas_ph_binary_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Pearce-Hall (PH) learning model for binary inputs.
%
% The PH model is described in:
%
% http://www.scholarpedia.org/article/Pearce-Hall_error_learning_theory,
%
% and in:
%
% Pearce, J. M., & Bouton, M. E. (2001). Theories of Associative
% Learning in Animals. Annual Review of Psychology, 52(1),
% 111–139. http://doi.org/10.1146/annurev.psych.52.1.111
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The PH configuration consists of the priors of the learning rate alpha and the initial value v_0
% of the value v. The priors are Gaussian in the space where the parameters they refer to are
% estimated. They are specified by their sufficient statistics: mean and variance (NOT standard
% deviation).
% 
% Both alpha and v_0 are estimated in 'logit-space' because they are bounded inside the unit
% interval.
%
% 'Logit-space' is a logistic sigmoid transformation of native space
% 
% tapas_logit(x) = ln(x/(1-x)); x = 1/(1+exp(-tapas_logit(x)))
%
% Any of the parameters can be fixed (i.e., set to a fixed value) by setting the variance of their
% prior to zero. To fix v_0 to 0.5 set the mean as well as the variance of the prior to zero.
%
% Fitted trajectories can be plotted by using the command
%
% >> tapas_ph_binary_plotTraj(est)
% 
% where est is the stucture returned by tapas_fitModel. This structure contains the estimated
% parameters al_0, v_0, and S in est.p_prc and the estimated trajectories of the agent's
% representations:
%              
%         est.p_prc.v_0      initial value of v
%         est.p_prc.al_0     initial value of alpha
%         est.p_prc.S        intensity of CS (cf. http://www.scholarpedia.org/article/Pearce
%                                                                -Hall_error_learning_theory)
%
%         est.traj.v         value: v
%         est.traj.al        associability: alpha (this is simply the absolute prediction error
%                                from the previous trial)
%         est.traj.da        prediction error: delta
%
% Tips:
% - Your guide to all these adjustments is the log-model evidence (LME). Whenever the LME increases
%   by at least 3 across datasets, the adjustment was a good idea and can be justified by just this:
%   the LME increased, so you had a better model.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2015 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Model name
c.model = 'ph_binary';

% Initial v
c.logitv_0mu = tapas_logit(0.5, 1);
c.logitv_0sa = 0;

% Initial alpha
c.logital_0mu = tapas_logit(0.5, 1);
c.logital_0sa = 1;

% S
c.logSmu = log(0.1);
c.logSsa = 8;

% Gather prior settings in vectors
c.priormus = [
    c.logitv_0mu,...
    c.logital_0mu,...
    c.logSmu,...
         ];

c.priorsas = [
    c.logitv_0sa,...
    c.logital_0sa,...
    c.logSsa,...
         ];

% Model function handle
c.prc_fun = @tapas_ph_binary;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_ph_binary_transp;

return;

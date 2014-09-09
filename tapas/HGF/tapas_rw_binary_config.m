function c = tapas_rw_binary_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Rescorla-Wagner (RW) learning model for binary inputs.
%
% The RW model was introduced in :
%
% Rescorla, R. A., and Wagner, A. R. (1972). "A theory of Pavlovian conditioning:
% Variations in the effectiveness of reinforcement," in Classical Conditioning
% II: Current Research and Theory, eds. A. H. Black and W. F. Prokasy (New
% York: Appleton-Century-Crofts), 64-99.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The RW configuration consists of the priors of the learning rate alpha and the initial value v_0
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
% >> tapas_rw_binary_plotTraj(est)
% 
% where est is the stucture returned by tapas_fitModel. This structure contains the estimated
% parameters alpha and v_0 in est.p_prc and the estimated trajectories of the agent's
% representations:
%              
%         est.p_prc.v_0      initial value of v
%         est.p_prc.alpha    alpha
%
%         est.traj.v         value: v
%         est.traj.da        prediction error: delta
%
% Tips:
% - Your guide to adjustments of the priors is the negative free energy F. Whenever F
%   increases by at least 3, the adjustment was a good idea and can be justified by just this: F
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
c.model = 'tapas_rw_binary';

% Initial v
c.logitv_0mu = tapas_logit(0.5, 1);
c.logitv_0sa = 0;

% Alpha
c.logitalmu = tapas_logit(0.5, 1);
c.logitalsa = 1;

% Gather prior settings in vectors
c.priormus = [
    c.logitv_0mu,...
    c.logitalmu,...
         ];

c.priorsas = [
    c.logitv_0sa,...
    c.logitalsa,...
         ];

% Model function handle
c.prc_fun = @tapas_rw_binary;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_rw_binary_transp;

return;

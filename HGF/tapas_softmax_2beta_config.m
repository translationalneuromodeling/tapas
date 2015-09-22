function c = tapas_softmax_2beta_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the softmax observation model for multinomial responses and
% different betas for rewards and punishments. NOTE: this model is acausal in situations where
% decisions are made before the outcome (reward or punishhment) is known.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Is the decision based on predictions or posteriors? Comment as appropriate.
c.predorpost = 1; % Predictions
%c.predorpost = 2; % Posteriors

% Model name
c.model = 'softmax_2beta';

% Sufficient statistics of Gaussian parameter priors

% Betas
c.logbemu = [log(1), log(1)];
c.logbesa = [   4^2,    4^2];

% Gather prior settings in vectors
c.priormus = [
    c.logbemu,...
         ];

c.priorsas = [
    c.logbesa,...
         ];

% Model filehandle
c.obs_fun = @tapas_softmax_2beta;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_softmax_2beta_transp;

return;

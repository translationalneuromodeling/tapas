function c = tapas_beta_obs_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the beta observation model for responses in the unit interval
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The parameter nu'("nu prime") regulates the width of the beta density. It is defined as nu' =
% nu - 2, with nu = alpha + beta, where alpha and beta are the conventional parameters of the
% beta density.
%
% nu' is estimated in log-space, thus enforcing a constraint of nu > 2. This ensures that the
% beta distributions underlying responses are sufficiently regular.
%
% nu can be interpreted as inverse decision noise. To have a shrinkage prior on this, choose a
% high value.
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
%c.predorpost = 1; % Predictions
c.predorpost = 2; % Posteriors

% Model name
c.model = 'beta_obs';

% Sufficient statistics of Gaussian parameter priors

% nu'
c.lognuprmu = log(128);
c.lognuprsa = 4;

% Gather prior settings in vectors
c.priormus = [
    c.lognuprmu,...
         ];

c.priorsas = [
    c.lognuprsa,...
         ];

% Model filehandle
c.obs_fun = @tapas_beta_obs;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_beta_obs_transp;

return;

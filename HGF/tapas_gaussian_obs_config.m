function c = tapas_gaussian_obs_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Gaussian noise observation model for continuous responses
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The Gaussian noise observation model assumes that responses have a Gaussian distribution around
% the inferred mean of the relevant state. The only parameter of the model is the noise variance
% (NOT standard deviation) zeta.
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
c.model = 'tapas_gaussian_obs';

% Sufficient statistics of Gaussian parameter priors
%
% Zeta
c.logzemu = log(0.005);
c.logzesa = 0.1;

% Gather prior settings in vectors
c.priormus = [
    c.logzemu,...
         ];

c.priorsas = [
    c.logzesa,...
         ];

% Model filehandle
c.obs_fun = @tapas_gaussian_obs;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_gaussian_obs_transp;

return;

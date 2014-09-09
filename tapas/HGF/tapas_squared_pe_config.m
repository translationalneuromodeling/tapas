function c = tapas_squared_pe_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuraton for the squared-prediction error optimization of perceptual parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The squared-prediction error optimization infers the perceptual parameter values that lead to the
% best predictions of input according to the (conditional) criterion of the sum of squared
% errors. The criterion is conditional in the sense that the priors on the perceptual parameters
% retain a certain weight determined by the parameter zeta, whose prior is defined below. Zeta can
% be interpreted as an inverse weight on the prediction errors: greater values of zeta lead to less
% influence of prediction errors as opposed to priors. One would usually leave zeta fixed.
%
% Usage:
%     tapas_fitModel([], inputs, '<perceptual_model>', 'tapas_squared_pe_config', ...)
% 
% Note that the first argument (responses) is empty.
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
c.logzemu = log(0.05);
c.logzesa = 0;

% Gather prior settings in vectors
c.priormus = [
    c.logzemu,...
         ];

c.priorsas = [
    c.logzesa,...
         ];

% Model filehandle
c.obs_fun = @tapas_squared_pe;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_squared_pe_transp;

return;

function c = tapas_bayes_optimal_categorical_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuraton for the estimation of Bayes optimal perceptual parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%     fitModel([], inputs, '<perceptual_model>', 'bayes_optimal_categorical_config', ...)
% 
% Note that the first argument (responses) is empty.
%
% This optimization requires no observation parameters. The corresponding variables are therefore
% empty.
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
c.model = 'Bayes optimal categorical';

% Gather prior settings in vectors
c.priormus = [];
c.priorsas = [];

% Model filehandle
c.obs_fun = @tapas_bayes_optimal_categorical;

% This is the handle to a dummy function since there are no parameters to transform
c.transp_obs_fun = @tapas_bayes_optimal_categorical_transp;

return;

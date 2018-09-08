function c = tapas_softmax_mu3_wld_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the softmax observation model for multinomial responses with phasic
% volatility exp(mu3) as the decision temperature and parameters accounting for win- and
% loss-distortion of state values.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2018 Christoph Mathys, TNU, UZH & ETHZ
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
c.model = 'softmax_mu3_wld';

% Sufficient statistics of Gaussian parameter priors

% Win-distortion
c.la_wdmu = 0;
c.la_wdsa = 2^-2;

% Loss-distortion
c.la_ldmu = 0;
c.la_ldsa = 2^-2;

% Gather prior settings in vectors
c.priormus = [
    c.la_wdmu,...
    c.la_ldmu,...
         ];

c.priorsas = [
    c.la_wdsa,...
    c.la_ldsa,...
         ];

% Model filehandle
c.obs_fun = @tapas_softmax_mu3_wld;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_softmax_mu3_wld_transp;

return;

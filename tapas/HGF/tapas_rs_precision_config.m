function c = tapas_rs_precision_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the 'precision' response model according to:
%
% Vossel, S.*, Mathys, C.*, Daunizeau, J., Bauer, M., Driver, J., Friston, K. J., and Stephan, K. E.
% (2013). Spatial Attention, Precision, and Bayesian Inference: A Study of Saccadic Response Speed.
% Cerebral Cortex.
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
c.model = 'Response speed: precision';

% Sufficient statistics of Gaussian parameter priors
%
% Zeta_1_valid
c.logze1vmu = log(0.0052);
c.logze1vsa = 0.1;

% Zeta_1_invalid
c.logze1imu = log(0.0052);
c.logze1isa = 0.1;

% Zeta_2
c.logze2mu = log(0.0006);
c.logze2sa = 0.001;

% Zeta_3
c.logze3mu = log(0.001);
c.logze3sa = 1000;

% Gather prior settings in vectors
c.priormus = [
    c.logze1vmu,...
    c.logze1imu,...
    c.logze2mu,...
    c.logze3mu,...
         ];

c.priorsas = [
    c.logze1vsa,...
    c.logze1isa,...
    c.logze2sa,...
    c.logze3sa,...
         ];

% Model filehandle
c.obs_fun = @tapas_rs_precision;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_rs_transp;

return;

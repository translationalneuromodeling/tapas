function c = tapas_logrt_linear_binary_minimal_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the linear log-reaction time response model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The Gaussian noise observation model assumes that responses have a Gaussian distribution around
% the inferred mean of the relevant state. The only parameter of the model is the noise variance
% (NOT standard deviation) zeta.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Model name
c.model = 'Minimal linear log-reaction time for binary models';

% Sufficient statistics of Gaussian parameter priors
%
% Beta_0
c.be0mu = log(500); 
c.be0sa = 4;

% Beta_1
c.be1mu = 0;
c.be1sa = 4;

% Beta_2
c.be2mu = 0; 
c.be2sa = 4;

% Zeta
c.logzemu = log(log(20));
c.logzesa = log(2);

% Gather prior settings in vectors
c.priormus = [
    c.be0mu,...
    c.be1mu,...
    c.be2mu,...
    c.logzemu,...
         ];

c.priorsas = [
    c.be0sa,...
    c.be1sa,...
    c.be2sa,...
    c.logzesa,...
         ];

% Model filehandle
c.obs_fun = @tapas_logrt_linear_binary_minimal;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_logrt_linear_binary_minimal_transp;

return;

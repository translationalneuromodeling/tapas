function c = tapas_condhalluc_obs2_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the response model used to analyze data from conditioned
% hallucination paradigm by Powers & Corlett
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The rationale for this model is as follows:
%
% TO BE DESCRIBED...
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Model name
c.model = 'tapas_condhalluc_obs2';

% Sufficient statistics of Gaussian parameter priors

% Beta
c.logbemu = log(48);
c.logbesa = 1;

% Nu
c.lognumu = log(1);
c.lognusa = 1;

% Gather prior settings in vectors
c.priormus = [
    c.logbemu,...
    c.lognumu,...
         ];

c.priorsas = [
    c.logbesa,...
    c.lognusa,...
         ];

% Model filehandle
c.obs_fun = @tapas_condhalluc_obs2;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_condhalluc_obs2_transp;

return;

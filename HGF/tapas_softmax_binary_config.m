function c = tapas_softmax_binary_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the softmax observation model for binary responses
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The binary softmax function is the logistic sigmoid
%
% f(x) = 1/(1+exp(-beta*(v1-v0)))
%
% where v1 and v0 are the values of options 1 and 0, respectively, and beta > 0 is a parameter that
% determines the slope of the sigmoid. Beta is sometimes referred to as the (inverse) decision
% temperature. In the formulation above, it represents the probability of choosing option 1.
% Reversing the roles of v1 and v0 yields the probability of choosing option 0.
%
% Beta can be interpreted as inverse decision noise. To have a shrinkage prior on this, choose a
% high value. It is estimated log-space since it has a natural lower bound at zero.
%
% In general, v1 and v0 can be any real numbers. In the context of a perceptual model, however, v1
% and v0 will be expected rewards, that is the product of an outcome probability and the reward
% entailed by that outcome. If outcomes 1 and 0 entail rewards r1 and r0, respectively, then v1 =
% mu1hat*r1 and v0 = (1-mu1hat)*r0 because mu1hat is the probability of outcome 1 and 1-mu1hat the
% probability of outcome 0.
%
% This observation model expects the first column of the input matrix to contain the outcomes: 1 or
% 0 for each trial. The SECOND COLUMN of the input matrix is expected to contain the rewards for
% OPTION 0 and the THIRD COLUMN the rewards for OPTION 1. If the input matrix contains only one
% column, the rewards are all assumed to be equal to 1.
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

% Is the decision based on predictions or posteriors? Comment as appropriate.
c.predorpost = 1; % Predictions
%c.predorpost = 2; % Posteriors

% Model name
c.model = 'tapas_softmax_binary';

% Sufficient statistics of Gaussian parameter priors

% Beta
c.logbemu = log(48);
c.logbesa = 1;

% Gather prior settings in vectors
c.priormus = [
    c.logbemu,...
         ];

c.priorsas = [
    c.logbesa,...
         ];

% Model filehandle
c.obs_fun = @tapas_softmax_binary;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_softmax_binary_transp;

return;

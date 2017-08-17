function c = tapas_condhalluc_obs_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the response model used to analyze data from conditioned
% hallucination paradigm by Powers & Corlett
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The rationale for this model is as follows:
%
% We apply decision noise (i.e., a logistic sigmoid, see below) to the probability that the subject
% says “yes” on a given trial:
%
% p( yes | belief ) = sigmoid( belief ),
%
% where belief = p( tone | percept, light ),
%
% and where in turn “percept” is the subjective experience of hearing (or not hearing) a tone, while
% “tone” is the objective presentation of a tone, and “light” is the presentation of a light.
%
% In trial where there is a tone, we may use Bayes’ theorem to get the belief:
%
% belief = p( tone | percept, light ) = p( percept | tone )*p( tone | light ) / (p( percept |
%                 tone )*p( tone | light ) + p( percept | no tone )*p( no tone | light ))
%
% Unpacking the various ingredients, we have
%
% - p( percept | tone ) is given by experimental design: the true positive rate of the tone
%   presented without light at each trial - 1/4, 1/2, or 3/4
%
% - p( tone | light ) is the prior from learning using the HGF: mu1hat
%
% - p( percept | no tone ) is the false positive rate, which we can take to be 1 - mu1hat
%
% - p( no tone | light ) is the other half of the prior: 1 - mu1hat
%
% In trials where there is no tone, the belief is mu1hat.
%
% The logistic sigmoid is
%
% f(x) = 1/(1+exp(-beta*(v1-v0))),
%
% where v1 and v0 are the values of options 1 and 0, respectively, and beta > 0 is a parameter that
% determines the slope of the sigmoid. Beta is sometimes referred to as the (inverse) decision
% temperature. In the formulation above, it represents the probability of choosing option 1.
% Reversing the roles of v1 and v0 yields the probability of choosing option 0.
%
% Beta can be interpreted as inverse decision noise. To have a shrinkage prior on this, choose a
% high value. It is estimated log-space since it has a natural lower bound at zero. In general, v1
% and v0 can be any real numbers.
%
% This observation model expects the first column of the input matrix to contain the outcomes
% (corresponding in this case to the choices of the subject): 1 or 0 for each trial. The SECOND
% COLUMN of the input matrix is expected to contain the true-positive rate of the stimulus for
% each trial. The response matrix only contains one column consisting of the choices of the
% subject. This means that it will be identical to the first column of the input matrix.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2015 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Model name
c.model = 'tapas_condhalluc_obs';

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
c.obs_fun = @tapas_condhalluc_obs;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_condhalluc_obs_transp;

return;

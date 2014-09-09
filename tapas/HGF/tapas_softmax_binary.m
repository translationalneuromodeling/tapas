function logp = tapas_softmax_binary(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the binary softmax model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zeta to its native space
be = exp(ptrans(1));

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
logp = NaN(length(infStates(:,1,1)),1);

% Check input format
if size(r.u,2) ~= 1 && size(r.u,2) ~= 3
    error('Inputs incompatible with tapas_softmax_binary observation model. See tapas_softmax_binary_config.m')
end

% Weed irregular trials out from inferred states, responses, and inputs
x = infStates(:,1,1);
x(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

if size(r.u,2) == 3
    r0 = r.u(:,2);
    r0(r.irr) = [];
    r1 = r.u(:,3);
    r1(r.irr) = [];
end

% If input matrix has only one column, assume the weight (reward value)
% of both options is equal to 1
if size(r.u,2) == 1
    % Calculate log-probabilities for non-irregular trials
    logp(not(ismember(1:length(logp),r.irr))) = -log(1+exp(-be.*(2.*x-1).*(2.*y-1)));
end
% If input matrix has three columns, the second contains the weights of
% outcome 0 and the third contains the weights of outcome 1
if size(r.u,2) == 3
    % Calculate log-probabilities for non-irregular trials
    logp(not(ismember(1:length(logp),r.irr))) = -log(1+exp(-be.*(r1.*x-r0.*(1-x)).*(2.*y-1)));
end

return;
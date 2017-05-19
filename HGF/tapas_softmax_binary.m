function [logp, yhat, res] = tapas_softmax_binary(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the binary softmax model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Predictions or posteriors?
pop = 1; % Default: predictions
if r.c_obs.predorpost == 2
    pop = 3; % Alternative: posteriors
end

% Transform zeta to its native space
be = exp(ptrans(1));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Check input format
if size(r.u,2) ~= 1 && size(r.u,2) ~= 3
    error('tapas:hgf:SoftMaxBinary:InputsIncompatible', 'Inputs incompatible with tapas_softmax_binary observation model. See tapas_softmax_binary_config.m.')
end

% Weed irregular trials out from inferred states, responses, and inputs
x = infStates(:,1,pop);
x(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

if size(r.u,2) == 3
    r0 = r.u(:,2);
    r0(r.irr) = [];
    r1 = r.u(:,3);
    r1(r.irr) = [];
end

% Calculate log-probabilities for non-irregular trials
% If input matrix has only one column, assume the weight (reward value)
% of both options is equal to 1
if size(r.u,2) == 1
    % Probability of observed choice
    probc = 1./(1+exp(-be.*(2.*x-1).*(2.*y-1)));
end
% If input matrix has three columns, the second contains the weights of
% outcome 0 and the third contains the weights of outcome 1
if size(r.u,2) == 3
    % Probability of observed choice
    probc = 1./(1+exp(-be.*(r1.*x-r0.*(1-x)).*(2.*y-1)));
end
reg = ~ismember(1:n,r.irr);
logp(reg) = log(probc);
yh = y.*probc +(1-y).*(1-probc);
yhat(reg) = yh;
res(reg) = (y -yh)./sqrt(yh.*(1 -yh));

return;

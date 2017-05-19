function [logp, yhat, res] = tapas_softmax_2beta(r, infStates, ptrans)
% Calculates the log-probability of responses under the softmax model with different betas for
% rewards and punishments
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013-2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Predictions or posteriors?
predorpost = r.c_obs.predorpost;

% Transform betas to their native space
% be(1): rewards, be(2): punishments
be = exp(ptrans(1:2));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from inferred states, inputs, and responses
states = squeeze(infStates(:,1,:,1,predorpost));
states(r.irr,:) = [];
u = r.u(:,1);
u(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

% Number of choices
nc = size(infStates,3);

% Partition functions
Z1 = sum(exp(be(1)*states),2);
Z1 = repmat(Z1,1,nc);

Z2 = sum(exp(be(2)*states),2);
Z2 = repmat(Z2,1,nc);

% Softmax probabilities
prob1 = exp(be(1)*states)./Z1;
prob2 = exp(be(2)*states)./Z2;

% Extract probabilities of chosen options
probc1 = prob1(sub2ind(size(prob1), 1:length(y), y'));
probc2 = prob2(sub2ind(size(prob2), 1:length(y), y'));

% Choose the correct column
probc = probc1'.*(u==1) +probc2'.*(u==0);

% Calculate log-probabilities for non-irregular trials
reg = ~ismember(1:n,r.irr);
logp(reg) = log(probc);
yhat(reg) = probc;
res(reg) = -log(probc);

return;

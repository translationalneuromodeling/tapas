function logp = tapas_softmax(r, infStates, ptrans)
% Calculates the log-probability of responses under the softmax model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform beta to its native space
be = exp(ptrans(1));

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
logp = NaN(length(infStates(:,1,1,1)),1);

% Weed irregular trials out from inferred states and responses
states = squeeze(infStates(:,1,:,1));
states(r.irr,:) = [];
y = r.y(:,1);
y(r.irr) = [];

% Number of choices
nc = size(infStates,3);

% Partition functions
Z = sum(exp(be*states),2);
Z = repmat(Z,1,nc);

% Softmax probabilities
prob = exp(be*states)./Z;

% Extract probabilities of chosen options
probc = prob(sub2ind(size(prob), 1:length(y), y'));

% Calculate log-probabilities for non-irregular trials
logp(not(ismember(1:length(logp),r.irr))) = log(probc);

return;

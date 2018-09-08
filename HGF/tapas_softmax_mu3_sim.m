function y = tapas_softmax_mu3_sim(r, infStates, p)
% Simulates observations from a Boltzmann distribution with volatility as temperature
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2017 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Number of choices
nc = size(infStates,3);

% The value of the last dimension of infStates determines whether responses are
% based on: 1 = predictions, 2 = posteriors.
states = squeeze(infStates(:,1,:,1));
mu3 = infStates(:,3,1,3);

% Inverse decision temperature
be = exp(-mu3);
be = repmat(be,1,nc);

% Partition functions
Z = sum(exp(be.*states),2);
Z = repmat(Z,1,nc);

% Softmax probabilities
prob = exp(be.*states)./Z;

% Initialize random number generator
rng('shuffle');

% Draw responses
n = size(infStates,1);
y = NaN(n,1);

for j=1:n
    y(j) = find(mnrnd(1, prob(j,:)));
end

return;

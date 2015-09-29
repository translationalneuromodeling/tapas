function y = tapas_softmax_sim(r, infStates, p)
% Simulates observations from a Bernoulli distribution
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Number of response options
nw = size(infStates,3);

% The value of the last dimension of infStates determines whether responses are
% based on: 1 = predictions, 2 = posteriors.
states = squeeze(infStates(:,1,:,1,2));
be = p;

% Partition functions
Z = sum(exp(be*states),2);
Z = repmat(Z,1,nw);

% Softmax probabilities
prob = exp(be*states)./Z;

% Draw responses
n = size(infStates,1);
y = NaN(n,1);

for j=1:n
    y(j) = find(mnrnd(1, prob(j,:)));
end

return;

function y = tapas_softmax_binary_sim(r, infStates, p)
% Simulates observations from a Bernoulli distribution
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

mu1hat = infStates(:,1,1);
be = p;

if size(mu1hat,2) == 1
    if ~any(mu1hat<0) && ~any(mu1hat>1)
        % Apply the logistic sigmoid to the inferred states
        prob = tapas_sgm(be.*(2.*mu1hat-1),1);
    else
        error('tapas:hgf:SoftMaxBinary:InfStatesIncompatible', 'infStates incompatible with tapas_softmax_binary observation model.')
    end
else
    % Apply the logistic sigmoid to the inferred states
    prob = tapas_sgm(be.*(mu1hat(:,1)-mu1hat(:,2)),1);
end

% Initialize random number generator
rng('shuffle');

% Simulate
y = binornd(1, prob);

return;

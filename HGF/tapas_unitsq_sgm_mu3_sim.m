function y = tapas_unitsq_sgm_mu3_sim(r, infStates, p)
% Simulates observations from a Bernoulli distribution
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2018 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

mu1hat = infStates(:,1,1);
mu3hat = infStates(:,3,1);
ze = exp(-mu3hat);

% Apply the unit-square sigmoid to the inferred states
prob = mu1hat.^ze./(mu1hat.^ze+(1-mu1hat).^ze);

% Initialize random number generator
if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end

% Simulate
y = binornd(1, prob);

return;

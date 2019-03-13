function y = tapas_beta_obs_sim(r, infStates, p)
% Simulates observations from a Bernoulli distribution
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2015 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Inferred states
mu = tapas_sgm(infStates(:,2,3), 1);
if strcmp(r.c_prc.model,'hgf_whichworld')
    mu = tapas_sgm(infStates(:,2,1,3), 1);
end
if strcmp(r.c_prc.model,'ph_binary')
    mu = infStates(:,2);
end

% Parameter nu-prime
nupr = p;

% Nu is nu-prime plus two (sometimes)
%nu = nupr+2;
nu = nupr;

% Calculate alpha and beta from mu and nu
al = mu.*nu;
be = nu - al;

% Initialize random number generator
if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end

% Simulate
y = betarnd(al, be);

return;

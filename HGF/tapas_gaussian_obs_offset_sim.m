function y = tapas_gaussian_obs_offset_sim(r, infStates, p)
% Simulates observations with Gaussian noise
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2018 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Get parameters
ze = p(1);
la = p(2);

% Get observation prediction trajectory
yhat = la + infStates(:,1,1);

% Number of trials
n = length(yhat);

% Initialize random number generator
if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end

% Simulate
y = yhat +sqrt(ze)*randn(n, 1);

return;

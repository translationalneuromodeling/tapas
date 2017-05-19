function [logp, yhat, res] = tapas_bayes_optimal_categorical(r, infStates, ptrans)
% Calculates the log-probability of the inputs given the current predictions
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from predictions
pred = squeeze(infStates(:,1,:,1));
pred(r.irr,:) = [];

% Weed irregular trials out from inputs
u = r.u(:,1);
u(r.irr) = [];

% Calculate probabilities of observed outcomes
for k = 1:length(u)
    p(k) = pred(k,u(k));
end

% Calculate log-probabilities for non-irregular trials
reg = ~ismember(1:n,r.irr);
logp(reg) = log(p);
yhat(reg) = p;
res(reg) = -log(p);

return;

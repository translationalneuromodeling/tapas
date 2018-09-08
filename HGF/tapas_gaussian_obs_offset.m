function [logp, yhat, res] = tapas_gaussian_obs_offset(r, infStates, ptrans)
% Calculates the log-probability of response y under the Gaussian noise model, with an offset
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2018 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zeta to its native space
ze = exp(ptrans(1));
la = ptrans(2);

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from inferred states and responses
x = la + infStates(:,1,1); % Add offset lambda
x(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
reg = ~ismember(1:n,r.irr);
logp(reg) = -1/2.*log(8*atan(1).*ze) -(y-x).^2./(2.*ze);
yhat(reg) = x;
res(reg) = y-x;

return;

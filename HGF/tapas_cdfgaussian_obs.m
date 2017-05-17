function [logp, yhat, res] = tapas_cdfgaussian_obs(r, infStates, ptrans)
% Calculates the log-probability of response y under a cumulative Gaussian distribution. This
% model has no free parameters.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2015 Christoph Mathys, TNU, UZH & ETHZ
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

% Weed irregular trials out from inferred states and responses
mu2 = infStates(:,2,3);
mu2(r.irr) = [];
sa2 = infStates(:,2,4);
sa2(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

% Probability mass for x2 < 0
x2lt0 = 0.5*(1 +erf((0 -mu2)./(sa2.*sqrt(2))));

% Probability of observed choice
probc = y.*(1 -x2lt0) +(1 -y).*x2lt0;

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
reg = ~ismember(1:n,r.irr);
logp(reg) = log(probc);
yh = 1 -x2lt0;
yhat(reg) = yh;
res(reg) = (y -yh)./sqrt(yh.*(1 -yh));

return;

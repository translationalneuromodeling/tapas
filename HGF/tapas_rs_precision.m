function [logp, yhat, res] = tapas_rs_precision(r, infStates, ptrans)
% Calculates the log-probability of response speed y (in units of ms^-1) according to the precision
% model introduced in:
%
% Vossel, S.*, Mathys, C.*, Daunizeau, J., Bauer, M., Driver, J., Friston, K. J., and Stephan, K. E.
% (2013). Spatial Attention, Precision, and Bayesian Inference: A Study of Saccadic Response Speed.
% Cerebral Cortex.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zetas to their native space
ze1v = exp(ptrans(1));
ze1i = exp(ptrans(2));
ze2  = exp(ptrans(3));
ze3  = exp(ptrans(4));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from inferred states, responses, and inputs
mu2hat = infStates(:,2,1);
mu2hat(r.irr) = [];

sa1hat = infStates(:,1,2);
pi1hat = 1./sa1hat;
pi1hat(r.irr) = [];

y = r.y(:,1);
y(r.irr) = [];

u = r.u(:,1);
u(r.irr) = [];

% Calculate alpha (i.e., attention)
alpha = tapas_sgm(sign(mu2hat).*(pi1hat-4),1);

% Calculate predicted response speed
rs = u.*(ze1v + ze2*alpha) + (1-u).*(ze1i + ze2*(1-alpha));

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
reg = ~ismember(1:n,r.irr)
logp(reg) = -1/2.*log(8*atan(1).*ze3) -(y-rs).^2./(2.*ze3);
yhat(reg) = rs;
res(reg) = y-rs;

return;

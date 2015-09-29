function sqpe = tapas_squared_pe(r, infStates, ptrans)
% Calculates squared prediction errors (pe) with zeta as a weight on pe's
% relative to perceptual priors
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zeta to its native space
ze = exp(ptrans(1));

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
sqpe = NaN(length(infStates(:,1,1)),1);

% Weed irregular trials out from inputs and predictions
%
% Inputs
u = r.u(:,1);
u(r.irr) = [];

% Predictions
mu1hat = infStates(:,1,1);
mu1hat(r.irr) = [];

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
sqpe(~ismember(1:length(sqpe),r.irr)) = -1/2.*log(8*atan(1).*ze) -(u-mu1hat).^2./(2.*ze);

return;

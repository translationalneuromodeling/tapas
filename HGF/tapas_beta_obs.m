function logp = tapas_beta_obs(r, infStates, ptrans)
% Calculates the log-probability of responses representing probabilities on the unit interval
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zeta to its native space
nupr = exp(ptrans(1));

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
logp = NaN(length(infStates(:,1,1)),1);

% Weed irregular trials out from inferred states and responses
mu = infStates(:,1);
if size(infStates,2) >= 3
    mu = mu + infStates(:,3);
end
if strcmp(r.c_prc.model,'hgf_whichworld')
    mu = infStates(:,1,1,3);
end

mu(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

% y has to be in the *open* unit interval
%y(y==0) = 1e-4;
%y(y==1) = 1-1e-4;
y = 0.95.*(y-0.5)+0.5; % Shrink all y values toward 1/2 by a factor of 0.95

% Nu is nu-prime plus two
%nu = nupr+2;
nu = nupr;

% Calculate alpha and beta from mu and nu
al = mu.*nu;
be = nu - al;

% Calculate log-probabilities for non-irregular trials
logp(not(ismember(1:length(logp),r.irr))) = log(betaDens(y,al,be));

end

function p = betaDens(x,alpha,beta)
% Check whether x is in the unit interval
if any(x(:)<0) || any(x(:)>1)
    error('tapas:hgf:BetaObs:ArgNotInUnitIntrv', 'Error: first argument to betaDens must be in the unit interval.');
end
% Check whether alpha and beta are greater than 0
if any(alpha(:)<=0) || any(beta(:)<=0)
    error('tapas:hgf:BetaObs:AlphaOrBetaNotPos', 'Error: alpha and beta have to be positive.');
end
% Calculate beta density
p = gamma(alpha+beta)./(gamma(alpha).*gamma(beta)).*x.^(alpha-1).*(1-x).^(beta-1);
end

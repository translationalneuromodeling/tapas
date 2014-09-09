function logp = tapas_unitsq_sgm_mod(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the unit-square 
%sigmoid model
%
% -----------------------------------------------------------------------------
% Copyright (C) 2012-2013 CM; 
% modified by aponteeduardo@gmail.com (2014.09.08), TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms 
% of the GNU General Public Licence (GPL), version 3. You can redistribute 
% it and/or modify it under the terms of the GPL (either version 3 or, at
% your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zeta to its native space
ze = exp(ptrans(1));

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
logp = NaN(length(infStates(:,1,1)),1);

% Weed irregular trials out from inferred states and responses
x = infStates(:,1,1);
x(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];

% Calculate log-probabilities for non-irregular trials

% Check if x is close to 0 or to 1. If that's the case be careful about
% computing the logs.

% Innefficient but safe and accurate

if any(abs(1-x) < 1e-4) || any(abs(x) < 1e-4)

    tlog = zeros(numel(x), 1);

    for i = 1:numel(x)
        if abs(1-x(i)) < 1e-3
            logx = log1p(x(i)-1);
            log1x = log(1-x(i));
        else
            logx = log(x(i));
            log1x = log1p(-x(i));
        end
        tlog(i) = y(i)*ze*(logx - log1x) + ze*log1x - ...
            log((1-x(i))^ze + x(i)^ze);
    end

else
    % All x are in a regime that will not produce degerate values.
    tlog = y.*ze.*(log(x) - log(1-x)) + ze.* log(1-x) - log((1-x).^ze + x.^ze);
end

logp(not(ismember(1:length(logp),r.irr))) = tlog;

if any(logp == -inf)
    error('Negative value in the log joint.')
end

end

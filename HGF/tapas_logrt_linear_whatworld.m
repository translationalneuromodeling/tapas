function [logp, yhat, res] = tapas_logrt_linear_whatworld(r, infStates, ptrans)
% Calculates the log-probability of log-reaction times y (in units of log-ms) according to the
% linear log-RT model developed with Louise Marshall and Sven Bestmann
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2014 Christoph Mathys, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zetas to their native space
be0  = ptrans(1);
be1  = ptrans(2);
be2  = ptrans(3);
be3  = ptrans(4);
ze   = exp(ptrans(5));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from responses and inputs
y = r.y(:,1);
y(r.irr) = [];

u = r.u(:,1);
u(r.irr) = [];

% Extract trajectories of interest from infStates
mu1hat = squeeze(infStates(:,1,:,:,1));
mu1    = squeeze(infStates(:,1,:,:,3));
mu2    = squeeze(infStates(:,2,:,:,3));
sa2    = squeeze(infStates(:,2,:,:,4));
mu3    = squeeze(infStates(:,3,1,1,3));

% Surprise
% ~~~~~~~~

% mu1 contains the actually occurring transition -> multiply with
% mu1hat to get probability of that transition (other elements are
% zero)
otp    = mu1.*mu1hat; % observed transition probabilities (3-dim)
otps3  = sum(otp, 3, 'omitnan');      % sum over 3rd dim
otps23 = sum(otps3, 2, 'omitnan');    % sum over 2nd dim

surp = -log(otps23);
surp(r.irr) = [];

% Expected uncertainty
% ~~~~~~~~~~~~~~~~~~~~
euo    = mu1.*sa2;    % expected uncertainty of observed transition (3-dim)
euos3  = sum(euo, 3, 'omitnan');      % sum over 3rd dim
euos23 = sum(euos3, 2, 'omitnan');    % sum over 2nd dim

to     = mu1.*mu2;    % tendency of observed transition (3-dim)
tos3   = sum(to, 3, 'omitnan');       % sum over 3rd dim
tos23  = sum(tos3, 2, 'omitnan');     % sum over 2nd dim

eu = tapas_sgm(tos23,1).*(1-tapas_sgm(tos23,1)).*euos23; % transform down to 1st level
eu(r.irr) = [];

% Unexpected uncertainty
% ~~~~~~~~~~~~~~~~~~~~~~
ueu = tapas_sgm(tos23,1).*(1-tapas_sgm(tos23,1)).*exp(mu3); % transform down to 1st level
ueu(r.irr) = [];

% Calculate predicted log-reaction time
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logrt = be0 +be1.*surp +be2.*eu +be3.*ueu;

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
reg = ~ismember(1:n,r.irr);
logp(reg) = -1/2.*log(8*atan(1).*ze) -(y-logrt).^2./(2.*ze);
yhat(reg) = logrt;
res(reg) = y-logrt;

return;

function [rpulset, pulset] = tapas_physio_filter_respiratory(...
    rpulset, rsampint, doNormalize)
% band-pass filters respiratory data (0.1...5 Hz)
%
%   rpulset = tapas_physio_filter_respiratory(pulset,rsampint)
%
% IN
%   rpulset
%   rsamping
%   doNormalize     default:false
%                   Optionally, data is normalized to be in -1...+1 range
%
% Author: Lars Kasper, 2011; heavily based on an earlier implementation of
% Chloe Hutton (FIL, UCL London)
%
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id$
if isempty(rpulset)
    rpulset = [];
    return;
end

% @SB: Can we not just replace all NaNs by 0s and take 0 as the offset, if it is first in time series?
% if rpulset has nans, subtraction and filtering will render whole time
% series to nan, so we replace them with zeros
% first, get value of first non-nan sample to correct offset
rpulsetFirst = find(~isnan(rpulset), 1, 'first');
rpulsetOffset = rpulset(rpulsetFirst);
% now, replace all nans with zeros
rpulsetNans = isnan(rpulset);
rpulset(rpulsetNans) = 0;

if nargin < 3
    doNormalize = true;
end

rpulset=rpulset-rpulsetOffset;

% bandpass filter
sampfreq    = 1/rsampint; % Hz

% Vlad: 0.09 0.7, 4th order butterworth, filtfilt for phase mod?
cutofflow   = 0.1; %10 seconds/rsampint units
cutoffhigh  = 5; %Hz
forder = 2;
[b,a] = butter(forder,2*[cutofflow, cutoffhigh]/sampfreq);

rpulset=filter(b,a,rpulset);

if doNormalize
    rpulset = rpulset/max(abs(rpulset));
end

% Now do a check for any outliers (> 3std)
mpulse=mean(rpulset);
stdpulse=std(rpulset);
outliers=find(rpulset>(mpulse+(3*stdpulse)));
pulset(outliers)=mpulse+(3*stdpulse);
outliers=find(rpulset<(mpulse-(3*stdpulse)));
rpulset(outliers)=mpulse-(3*stdpulse);
end

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
% $Id: tapas_physio_filter_respiratory.m 645 2015-01-15 20:41:00Z kasperla $
if isempty(rpulset)
    rpulset = [];
    return;
end

if nargin < 3
    doNormalize = true;
end

rpulset=rpulset-rpulset(1);

% bandpass filter
sampfreq    = 1/rsampint; % Hz
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

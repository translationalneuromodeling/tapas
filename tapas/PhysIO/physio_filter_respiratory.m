function [rpulset, pulset] = physio_filter_respiratory(rpulset,rsampint)
% high-pass filters respiratory data.
%
%   rpulset = physio_filter_respiratory(pulset,rsampint)
%
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
% $Id: physio_filter_respiratory.m 173 2013-04-03 10:55:39Z kasperla $

rpulset=rpulset-rpulset(1);

% bandpass filter
sampfreq    = 1/rsampint; % Hz
cutofflow   = 0.1; %10 seconds/rsampint units
cutoffhigh  = 5; %Hz
forder = 2;
[b,a] = butter(forder,2*[cutofflow, cutoffhigh]/sampfreq);

rpulset=filter(b,a,rpulset);

% Now do a check for any outliers (> 3std)
mpulse=mean(rpulset);
stdpulse=std(rpulset);
outliers=find(rpulset>(mpulse+(3*stdpulse)));
pulset(outliers)=mpulse+(3*stdpulse);
outliers=find(rpulset<(mpulse-(3*stdpulse)));
rpulset(outliers)=mpulse-(3*stdpulse);
end

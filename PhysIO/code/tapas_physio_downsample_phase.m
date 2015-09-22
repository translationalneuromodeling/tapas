function dphase = tapas_physio_downsample_phase(tphase, phase, tsample, rsampint)
% Author: Lars Kasper, using code from Chloe Hutton (FIL, UCL London)
%  
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_downsample_phase.m 354 2013-12-02 22:21:41Z kasperla $

n = zeros(size(tsample));
for t=1:length(tsample)
    n(t)=find(abs(tphase-tsample(t))<rsampint, 1, 'first');
end
dphase = phase(n);
end

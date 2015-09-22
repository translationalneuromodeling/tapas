function y = tapas_physio_maxfilter(x, n)
% Computes maximum over sliding window as a filtered version of the input
%
%   y = tapas_physio_maxfilter(x, n)
%
% IN
%   x   signal
%   n   number of samples in sliding window
%
% OUT
%   y   max-filtered signal
%
% EXAMPLE
%   tapas_physio_maxfilter
%
%   See also
%
% Author: Lars Kasper
% Created: 2015-01-11
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_maxfilter.m 638 2015-01-11 13:20:33Z kasperla $

nSamples = numel(x);
if nargin < 2
    n = ceil(nSamples/10);
end

y = zeros(size(x));

y(1) = max(x(1:n));

for iSample = 2:nSamples-n+1
    
    % if former maximum is not within n-range anymore, recalc max of range
    if y(iSample-1) == x(iSample-1)
        y(iSample) = max(x(iSample+(0:n-1)));
    else
        % max is either existing max or new value at end of n-range
        y(iSample) = max(y(iSample-1), x(iSample+n-1));
    end

end

y(nSamples-n+1:end) = max(x(end-n+1:end));
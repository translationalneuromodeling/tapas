function p = tapas_physio_prctile(x, percentile)
%returns the approximate value corresponding to a percentile for a given data vector
%
%   p = tapas_physio_prctile(x, percentile)
%
%   uses a sorting/interpolation approach as the original perctile in the
%   Matlab Statistics Toolbox
%
% IN
%   x           [N,1] data vector 
%   percentile  percentile which shall be queried (in percent)
% OUT
%   p           value of percentile for data vector
% EXAMPLE
%   x = randn(1000,1);
%   p = tapas_physio_prctile (x, 50) % returns median
%   figure;hist(x,100);yl = ylim;hold all;stem(p, yl(2));
%
%   See also prctile tapas_physio_plot_raw_physdata_diagnostics tapas_physio_create_scan_timing_from_gradients_philips
%
% Author: Lars Kasper
% Created: 2013-03-13
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_prctile.m 524 2014-08-13 16:21:56Z kasperla $
sx = sort(x);
N = length(x);

valP = round(((1:N)-0.5))/N*100;
iP = find(valP>percentile, 1, 'first');

switch iP
    case 1
        p = sx(1);
    otherwise
       p = (valP(iP)-percentile)/(valP(iP)-valP(iP-1))*sx(iP) + ...
           (percentile - valP(iP-1))/(valP(iP)-valP(iP-1))*sx(iP-1);
end

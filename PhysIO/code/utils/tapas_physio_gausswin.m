function w = tapas_physio_gausswin(L, a)
% vector of discrete Gaussian window 
%
%    w = tapas_physio_gausswin(L, a)
%
% IN
%   L           number of sampling points of gaussian window
%   a           alpha parameter: 1/std of gaussian, measures width of
%               fourier transform of gausswin
%               The larger a, the narrower the window
%               default: 2.5
% OUT
%   w           [L,1] vector of Gaussian window
% EXAMPLE
%   tapas_physio_suptitle
%
%   See also
 
% Author:   Lars Kasper
% Created:  2019-02-01
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

if nargin < 2
    a = 2.5;
end

N = L-1;
n = (0:N)'-N/2;
w = exp(-2*(a*n/N).^2);

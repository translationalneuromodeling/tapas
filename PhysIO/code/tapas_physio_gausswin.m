function w = tapas_physio_gausswin(L, a)
% Fast version of gausswin, without all checks
%
%    w = tapas_physio_gausswin(L, a)
%
% Matlab's "GAUSSWIN(N) returns an N-point Gaussian window.
%
%   GAUSSWIN(N, ALPHA) returns the ALPHA-valued N-point Gaussian
%   window.  ALPHA is defined as the reciprocal of the standard
%   deviation and is a measure of the width of its Fourier Transform.
%   As ALPHA increases, the width of the window will decrease. If omitted,
%   ALPHA is 2.5."
%
% Computed according to
%     [1] fredric j. harris [sic], On the Use of Windows for Harmonic
%         Analysis with the Discrete Fourier Transform, Proceedings of
%         the IEEE, Vol. 66, No. 1, January 1978

% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_gausswin
%
%   See also
%
% Author: Lars Kasper
% Created: 2017-11-16
% Copyright (C) 2017 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 775 2015-07-17 10:52:58Z kasperla $

if nargin < 2
    a = 2.5;
end

% Compute window according to [1]
N = L-1;
n = (0:N)'-N/2;
w = exp(-(1/2)*(a*n/(N/2)).^2);

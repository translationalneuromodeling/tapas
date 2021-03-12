function Y = tapas_uniqc_create_shepp_logan_4d()
% Creates 4D Shepp-Logan Phantom for tapas_uniqc_slider4d demo
%
%   output = tapas_uniqc_create_shepp_logan_4d(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_create_shepp_logan_4d
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-11-20
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% parameter specs, create shepp logan phantom
nX = 64;
nY = nX;
nSli = 32;
nDyn = 20;

P = phantom('Modified Shepp-Logan',nX);


% create different slices via shift
Y = zeros(nX,nY,nSli,1);
for iSli = 1:nSli
    Y(:,:,iSli,1) = circshift(P, iSli*floor(nX/nSli));
end

% replicate over number of dynamics and add some noise to make them
% different
Y = repmat(Y, [1 1 1 nDyn]) + 0.05*max(Y(:))*randn(nX,nY,nSli,nDyn);

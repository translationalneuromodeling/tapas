function otherImage = unwrap(this, applicationDimension)
% Unwraps phase of images along specified dimension
%(or assumes phase image as input for real valued data)
%
%   Y = MrImage()
%   Y.unwrap(inputs)
%
% This is a method of class MrImage.
%
% IN
%   applicationDimension    along which unwrapping is performed
%                           default: 4 (time)
% OUT
%
% EXAMPLE
%   unwrap
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-12-13
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 2
    applicationDimension = this.dimInfo.nDims;
end

% enforce separate call to unwrap for each 1D vector (e.g. time)
doApplicationLoopExplicitly = true;

if isreal(this)
    otherImage = this.perform_unary_operation(@unwrap, applicationDimension, ...
        doApplicationLoopExplicitly);
else
    otherImage = this.perform_unary_operation(@(x) unwrap(angle(x)), applicationDimension, ...
        doApplicationLoopExplicitly);
end

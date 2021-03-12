function iLastDimension = ndims(this)
% Returns Number of last non-singleton data dimension of image
%
% NOTE: For a [nX,nY,1,nVolumes], this would return 4, and is thus
% different from Matlab ndims
%
%   Y = MrImage()
%   iLastDimension = Y.ndims()
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y = MrImage(rand(128,128,33,50));
%   ndims(Y) % returns 4
%   Y = MrImage(rand(128,128,1,50));
%   ndims(Y) % returns 4
%   Y = MrImage(rand(128,128,50,1));
%   ndims(Y) % returns 3
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


iLastDimension = this.dimInfo.nDims;
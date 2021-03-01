function outputImage = flipud(this)
% Flips all slices Up-Down; mimicks flipud in matlab functionality
%
%   Y = MrImage()
%   Y.flipud(K)
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%   this    MrImage where data matrix is flipped and header is updated to
%           reflect that change
%
% EXAMPLE
%   Y = MrImage();
%   Y.flipud
%
%   See also MrImage flipud perform_unary_operation

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-08-04
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

outputImage = this.flip(1);
end
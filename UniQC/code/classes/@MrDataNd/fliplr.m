function outputImage = fliplr(this)
% Flips all slices Right-Left; mimicks fliplr in matlab functionality
%
%   Y = MrDataNd()
%   Y.fliplr(K)
%
% This is a method of class MrDataNd.
%
% IN
%
% OUT
%   this    MrDataNd where data matrix is flipped and dimInfo is updated to
%           reflect that change
%
% EXAMPLE
%   Y = MrDataNd();
%   Y.fliplr
%
%   See also MrDataNd fliplr perform_unary_operation

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

outputImage = this.flip(2);
end
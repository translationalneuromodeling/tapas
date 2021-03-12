function otherImage = rot90(this, K)
% (Multiple of) 90 deg image rotation; mimicks rot90 in matlab functionality
%
%   Y = MrDataNd()
%   Y.rot90(K)
%
% This is a method of class MrDataNd.
%
% IN
%   K   multiple of 90 degrees used for counterclockwise rotation 
%       i.e.    K = 0,1,2 (counterclockwise rotation) 
%           or  K = -1,-2,-3,... (clockwise rotation);
%       default : 1
% 
% OUT
%   otherImage      MrDataNd where data matrix is rotated and header is updated to
%                   reflect that change
%
% EXAMPLE
%   Y = MrDataNd();
%   Y.rot90(1); % rotate all slices counterclockwise by 90 degrees
%   Y.rot90(-2) % rotate all slices clockwise by 2*90 = 180 degrees
%
%   See also MrDataNd categorical/rot90

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

if nargin < 2
    K = 1;
end

otherImage = this.perform_unary_operation(@(x) rot90(x, K));

% First and second dimension change through rotation around 1, 3 etc.
% multiples of 90 degree...
% doSwapDimensions = mod(K,2) == 1;
% TODO: or shall this be reflected in affineTransformation?
% No, not in affine transformation! We rather take the stance that this is really
% a change of the data and if one wants to adapt the dimInfo, she has to do
% it actively.
doSwapDimensions = false;

if doSwapDimensions  
    otherImage.dimInfo.permute([2 1]);
end
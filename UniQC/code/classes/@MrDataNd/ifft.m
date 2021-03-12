function outputImage = ifft(this, applicationDimension)
% IFFT (including IFFT-shift and normalization) using image2k
%
%   Y = MrImage()
%   Y.ifft(applicationDimension)
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y = MrImage();
%   Y.ifft(4); % convert each voxel's time series to frequency space by
%               applying iFFT to 4th dimension
%   Y.ifft('2D') % slice-wise iFFT for transversal slices
%   Y.ifft([1 2]) % slice-wise iFFT, same as previous
%   Y.ifft([2 3]) % slice-wise iFFT, but for sagittal slices (dim 2 and 3 = 1
%                %  slice)
%   Y.ifft('3D') % volume-wise iFFT
%   Y.ifft([1 2 3]) % volume-wise iFFT, same as previous
%   Y.ifft([1 2 4]) % volume-wise iFFT, but for time series of a k-space
%                  % voxel, acquired in a 2D k-space
%            
%
%   See also MrImage MrImage.image2k

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-12-12
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

outputImage = image2k(this, applicationDimension);
function outputImage = fft(this, applicationDimension)
% FFT (including FFT-shift and normalization) using k2image
%
%   Y = MrImage()
%   Y.fft(applicationDimension)
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y = MrImage();
%   Y.fft(4); % convert each voxel's time series to frequency space by
%               applying FFT to 4th dimension
%   Y.fft('2D') % slice-wise FFT for transversal slices
%   Y.fft([1 2]) % slice-wise FFT, same as previous
%   Y.fft([2 3]) % slice-wise FFT, but for sagittal slices (dim 2 and 3 = 1
%                %  slice)
%   Y.fft('3D') % volume-wise FFT
%   Y.fft([1 2 3]) % volume-wise FFT, same as previous
%   Y.fft([1 2 4]) % volume-wise FFT, but for time series of a k-space
%                  % voxel, acquired in a 2D k-space
%            
%
%   See also MrImage MrImage.k2image

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

outputImage = k2image(this, applicationDimension);
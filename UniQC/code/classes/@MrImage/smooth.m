function outputImage = smooth(this, varargin)
% smoothes ND MrImage using Gaussian kernel and via SPM functionality
%
%   Y = MrImage()
%   sY = Y.smooth('fwhm', 5, 'splitDimLabels', 'echo', 'splitComplex', 'ri')
%
% This is a method of class MrImage.
%
% IN
%   fwhm            Full-width-at-half-maximum of the smoothing kernel in
%                   millimeter (scalar or [1 x 3] array).
%                   Note that a scalar input is automatically extended.
%                   default: 8
%
%   splitDimLabels  Defines the dimensions along which the nD image should
%                   be split into 4D images as input for SPM smooth.
%                   Note that the smoothing is performed along the first
%                   three dimensions.
%                   default: all but {'x','y','z',t'}
%
%   splitComplex    'ri' or 'mp'
%                   If the data are complex numbers, real and imaginary or
%                   magnitude and phase need to be smoothed separately.
%                   default: ri (real and imaginary)
%                            makes most sense, because phase wraps would
%                            otherwise be smoothed over
%
% OUT
%
%   outputImage     MrImage with smoothing kernel applied
%
% EXAMPLE
%   smooth
%
%   See also MrImage MrImageSpm4D.smooth MrImage.split_complex
%   MrImage.apply_spm_method_per_4d_split

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-04-26
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

outputImage = this.copyobj;

%% Decide upon which type of ND image this is
defaults.fwhm            = 8;
defaults.splitDimLabels  = {};
defaults.splitComplex    = 'ri';

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

isComplexImage = ~isreal(outputImage);

if isComplexImage
    outputImage = outputImage.split_complex(splitComplex);
end

outputImage = outputImage.apply_spm_method_per_4d_split(@smooth, ...
    'methodParameters', {{'fwhm', fwhm}}, ...
    'splitDimLabels', splitDimLabels);

%% reassemble complex smoothed images into one again
if isComplexImage
  outputImage = outputImage.combine_complex();
end

end
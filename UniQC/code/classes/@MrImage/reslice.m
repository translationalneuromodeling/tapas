function outputImage = reslice(this, targetGeometry, varargin)
% Reslices nD MrImage in 4D chuncks via SPM functionality
%
%   Y = MrImage()
%   rY = Y.reslice(targetGeometry, 'splitDimLabels', ...
%   'echo', 'splitComplex', 'ri')
%
% This is a method of class MrImage.
%
% IN
%   targetGeometry  object of MrImageGeometry or MrImage
%                   Image will be resliced to this geometry
%
%   interpolation   degree of b-spline interpolation for estimation and reslicing
%                   default: 7
%
%   wrapping        fold-over direction (phase encode)
%                   default: [0 0 0] % none
%
%   masking         mask incomplete timeseries?
%                   default: true
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
% OUT
%
%   reslicedImage
%
% EXAMPLE
%   Y = MrImage();
%   Z = MrImage();
%   Y.reslice(Y, 'interpolation' 2)
%
%   See also MrImage MrImageGeometry spm_reslice spm_run_coreg
%   MrImageSpm4D.reslice MrImage/demo_reslice

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-27
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

%% copy object
outputImage = this.copyobj();

%% Parse input parameters and prepare image
spmDefaults.interpolation = 7;     % degree of b-spline interpolation for estimation and reslicing
spmDefaults.wrapping = [0 0 0];    % fold-over direction (phase encode)
spmDefaults.masking = 1;           % mask incomplete timeseries?
defaults.splitDimLabels = {};
defaults.splitComplex = 'ri';

[args, unusedVarargin] = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

methodParameters = {tapas_uniqc_propval(unusedVarargin, spmDefaults)};

% default split is along any dimensions other than {x, y, z, y}
if isempty(splitDimLabels)
    dimLabelsSpm4D = {'x','y','z','t'};
    splitDimLabels = setdiff(outputImage.dimInfo.dimLabels, dimLabelsSpm4D);
end

isComplexImage = ~isreal(outputImage);

if isComplexImage
    outputImage = outputImage.split_complex(splitComplex);
end

outputImage = outputImage.apply_spm_method_per_4d_split(...
    @(x, y) reslice(x, targetGeometry, y), ...
    'methodParameters', methodParameters, ...
    'splitDimLabels', splitDimLabels);

%% reassemble complex resliced images into one again
if isComplexImage
    outputImage = outputImage.combine_complex();
end

end

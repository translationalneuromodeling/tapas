function varargout = hist(this, nBins, varargin)
% Computes and plots histogram of image intensities
%
%   Y = MrImage()
%   [pixelsPerBin, figureHandle] = ...
%       Y.hist(nBins, 'ParameterName1', 'ParameterValue1', ...)
%
% This is a method of class MrImage.
%
% IN
%   nBins       number of histogram bins; default: 10
%   varargin    parameterName/Value pairs for selection of volumes/slices
%
% OUT
%
% EXAMPLE
%   Y.hist(100, 'z', 1, 't', 3:100, ...,
%           'x', 55:75)
%
%   See also MrImage

% Author:   Lars Kasper
% Created:  2015-08-23
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
    nBins = 10;
end

if nargin < 3
    imgSelect = this;
else
    imgSelect = this.select(varargin{:});
end

[pixelsPerBin, binPositions] = hist(imgSelect.data(:), nBins);

nPixels         = prod(imgSelect.dimInfo.nSamples);
nameString      = tapas_uniqc_str2label(sprintf('Pixel Histogram of %s (total: %d pixels)', ...
    this.name, nPixels));
figureHandle    = figure('Name', nameString);

bar(binPositions, pixelsPerBin/nPixels*100);
xlabel('Pixel intensity');
ylabel('Relative pixel count (%)');
title(nameString);

title(nameString);

if nargout
    varargout{1} = pixelsPerBin;
end

if nargout >=2 
    varargout{2} = figureHandle;
end
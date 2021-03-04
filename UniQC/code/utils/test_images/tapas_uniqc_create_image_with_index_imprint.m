function X = tapas_uniqc_create_image_with_index_imprint(X)
% Creates image with imprinted index of 3rd..nth dimension on each 2D slice
% (1st and 2nd dim are considered a slice)
%
%    X = tapas_uniqc_create_image_with_index_imprint(X)
%
%   NOTE: This function needs Matlab's Computer Vision System Toolbox for
%         insertText
%
% IN
%   X   n-dimensional image array or [1,nDims] vector of nSamples per
%   dimension
%
% OUT
%
% EXAMPLE
%   % puts slice and volume index on 4D random image
%   myImage = rand(64, 64, 20, 100);
%   tapas_uniqc_create_image_with_index_imprint(myImage);
%
%   % creates zero image with overlay of indces as pixel matrix
%   nSamples = [64, 64, 20, 100]
%   tapas_uniqc_create_image_with_index_imprint(nSamples);
%
%   See also insertText

% Author:   Lars Kasper
% Created:  2016-01-31
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 1
    X = [64, 64, 20, 100];
end

sizeImage = size(X);
nDims = sum(sizeImage>1);

% 1 dimensional input is considered a vector of sample numbers
isSizeVector = nDims == 1;

% nSamples transformed into image of zeros
if isSizeVector
    sizeImage = X;
    X = zeros(sizeImage);
    nDims = sum(sizeImage>1);
end


nSlices = numel(X(1,1,:));

% holds index for each dimension but the first two
subIndices = cell(1,nDims-2);

% For each slice (i.e. 3rd..nth dimension collapsed), determine sub-index
% (i.e. index for each dimension)
for iSlice = 1:nSlices
    adjustedFontSize = max(5, round(10/64*sizeImage(1)));
    [subIndices{:}] = ind2sub(sizeImage(3:end), iSlice);
    X(:,:,iSlice) = rgb2gray(insertText(X(:,:,iSlice), [1 1], ...
        num2str(cell2mat(subIndices)), ...
        'FontSize', adjustedFontSize, 'BoxOpacity', .9));
end
